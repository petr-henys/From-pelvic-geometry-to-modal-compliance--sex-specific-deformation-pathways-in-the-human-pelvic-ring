"""Build FEBio .feb file with material regions and mirrored ligament sets."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pyvista as pv
from dolfinx import plot as dplot
from scipy.spatial import KDTree

from config import SimulationConfig
from simulation.febio_parser import FEBio2Dolfinx
from simulation.utils import find_symmetry_plane


TETRAHEDRAL_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)


def write_febio_xml(
    output_path: Path,
    coords: np.ndarray,
    conn: np.ndarray,
    material_map: np.ndarray,
    region_names: dict[int, str],
    region_materials: dict[str, dict[str, float]],
    discrete_sets: dict[str, np.ndarray],
    surface_sets: dict[str, np.ndarray],
) -> None:
    """Write FEBio 4.0 XML file with mesh, materials, discrete sets, and surfaces."""
    root = ET.Element("febio_spec", attrib={"version": "4.0"})
    ET.SubElement(root, "Module", attrib={"type": "solid"})

    # Globals
    globals_el = ET.SubElement(root, "Globals")
    const = ET.SubElement(globals_el, "Constants")
    ET.SubElement(const, "T").text = "0"
    ET.SubElement(const, "P").text = "0"
    ET.SubElement(const, "R").text = "8.31446"
    ET.SubElement(const, "Fc").text = "96485.3"

    # Materials
    mat_el = ET.SubElement(root, "Material")
    for midx, (name, props) in enumerate(region_materials.items(), start=1):
        m = ET.SubElement(
            mat_el,
            "material",
            attrib={"id": str(midx), "name": name, "type": "isotropic elastic"},
        )
        ET.SubElement(m, "density").text = "1"
        ET.SubElement(m, "E").text = str(props["E"])
        ET.SubElement(m, "v").text = str(props["v"])

    # Mesh: nodes
    mesh_el = ET.SubElement(root, "Mesh")
    nodes_el = ET.SubElement(mesh_el, "Nodes", attrib={"name": "AllNodes"})
    for i, (x, y, z) in enumerate(coords, start=1):
        ET.SubElement(nodes_el, "node", attrib={"id": str(i)}).text = f"{x},{y},{z}"

    # Mesh: elements per region
    unique_regions = np.unique(material_map)
    for rid in unique_regions:
        part_name = region_names[rid]
        region_cells = np.where(material_map == rid)[0]
        elems_el = ET.SubElement(mesh_el, "Elements", attrib={"type": "tet4", "name": part_name})
        for j, cell_idx in enumerate(region_cells, start=1):
            a, b, c, d = conn[cell_idx] + 1
            ET.SubElement(elems_el, "elem", attrib={"id": str(j)}).text = f"{a},{b},{c},{d}"

    # Discrete sets
    for set_name, pairs in discrete_sets.items():
        ds_el = ET.SubElement(mesh_el, "DiscreteSet", attrib={"name": set_name})
        for a, b in pairs:
            ET.SubElement(ds_el, "delem").text = f"{a},{b}"

    # Surfaces
    for surf_name, tris in surface_sets.items():
        surf_el = ET.SubElement(mesh_el, "Surface", attrib={"name": surf_name})
        for tidx, (i, j, k) in enumerate(tris, start=1):
            ET.SubElement(surf_el, "tri3", attrib={"id": str(tidx)}).text = f"{i},{j},{k}"

    # MeshDomains
    md_el = ET.SubElement(root, "MeshDomains")
    for rid in unique_regions:
        part_name = region_names[rid]
        ET.SubElement(md_el, "SolidDomain", attrib={"name": part_name, "mat": part_name})

    # Discrete materials
    if discrete_sets:
        disc_el = ET.SubElement(root, "Discrete")
        for didx, set_name in enumerate(discrete_sets.keys(), start=1):
            dm = ET.SubElement(
                disc_el,
                "discrete_material",
                attrib={"id": str(didx), "name": set_name, "type": "linear spring"},
            )
            ET.SubElement(dm, "E").text = "1"
        for didx, set_name in enumerate(discrete_sets.keys(), start=1):
            ET.SubElement(
                disc_el, "discrete", attrib={"dmat": str(didx), "discrete_set": set_name}
            )

    # Output
    out_el = ET.SubElement(root, "Output")
    pf = ET.SubElement(out_el, "plotfile", attrib={"type": "febio"})
    for var in ("displacement", "stress", "relative volume"):
        ET.SubElement(pf, "var", attrib={"type": var})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="ISO-8859-1", xml_declaration=True)


def connected_components(conn: np.ndarray, cells: np.ndarray) -> list[np.ndarray]:
    """Find connected components in tetrahedral mesh."""
    cells = np.asarray(cells, dtype=int)
    if cells.size == 0:
        return []

    adjacency: dict[int, set[int]] = {int(c): set() for c in cells}
    face_owner: dict[tuple[int, int, int], int] = {}

    for cell in cells:
        nodes = conn[int(cell)]
        for pattern in TETRAHEDRAL_FACES:
            face = tuple(sorted(nodes[pattern]))
            owner = face_owner.get(face)
            if owner is None:
                face_owner[face] = int(cell)
            elif owner != cell:
                adjacency[int(cell)].add(owner)
                adjacency[owner].add(int(cell))

    visited: set[int] = set()
    components: list[np.ndarray] = []

    for cell in cells:
        icell = int(cell)
        if icell in visited:
            continue
        stack = [icell]
        visited.add(icell)
        comp = []
        while stack:
            current = stack.pop()
            comp.append(current)
            for nbr in adjacency[current]:
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        components.append(np.asarray(comp, dtype=int))

    return components
def identify_material_regions(
    grid: pv.UnstructuredGrid,
    conn: np.ndarray,
    template: pv.PolyData,
    normal: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
    span: float,
) -> tuple[np.ndarray, dict[int, str]]:
    """Identify bone (0), left SIJ cartilage (1), right SIJ cartilage (2), symphysis (3).

    Robustness improvements:
    - Use signed distance to the closed template surface (vtkImplicitPolyDataDistance)
      instead of select_enclosed_points, which is sensitive to triangle orientation
      and tiny numerical gaps. Cells are classified as bone when their centroid lies
      inside or within a small tolerance band around the surface.
    - Classify left/right by signed distance to the symmetry plane directly, with
      clear tie-breaking for the symphysis component (closest-to-plane by |d|).
    """
    surface = template.extract_surface().triangulate()
    centers = grid.cell_centers().points

    # Compute signed distance to the surface using VTK's implicit distance
    try:
        from vtkmodules.vtkCommonDataModel import vtkImplicitPolyDataDistance  # type: ignore
    except ImportError:  # pragma: no cover - vtk import guard
        # Fallback to PyVista's enclosure test if VTK import fails
        enclosed = pv.PolyData(centers).select_enclosed_points(surface, tolerance=tolerance)
        inside_mask = enclosed.point_data["SelectedPoints"].astype(bool)
    else:
        ipd = vtkImplicitPolyDataDistance()
        ipd.SetInput(surface)
        # Evaluate signed distance for each cell center
        signed_dist = np.array([ipd.EvaluateFunction(c) for c in centers], dtype=float)
        # Inside is negative for a properly oriented closed surface; include a tolerance band
        band = max(float(tolerance), 0.0025 * float(span))  # ~0.25% of bbox diagonal
        inside_mask = signed_dist <= band

    non_bone_indices = np.flatnonzero(~inside_mask)

    material_map = np.zeros(grid.n_cells, dtype=int)
    region_names: dict[int, str] = {0: "PelvisBone"}

    if non_bone_indices.size == 0:
        return material_map, region_names

    # Find connected components outside bone
    components = connected_components(conn, non_bone_indices)
    volumes_all = grid.compute_cell_sizes().cell_data["Volume"]

    # Compute component info
    component_info: list[dict] = []
    normal_unit = normal / np.linalg.norm(normal)
    for comp_cells in components:
        pts = centers[comp_cells]
        centroid = pts.mean(axis=0)
        distance = float(np.dot(centroid - origin, normal_unit))
        volume = float(volumes_all[comp_cells].sum())
        component_info.append(
            {"cells": comp_cells, "centroid": centroid, "distance": distance, "volume": volume}
        )

    # Filter by volume
    max_volume = max(info["volume"] for info in component_info)
    volume_threshold = max(max_volume * 1e-2, 1e-9)
    component_info = [info for info in component_info if info["volume"] >= volume_threshold]
    component_info.sort(key=lambda item: item["volume"], reverse=True)

    if len(component_info) != 3:
        raise RuntimeError(f"Expected 3 non-bone components, found {len(component_info)}")

    # Identify symphysis (closest to plane; tie-break by largest volume)
    symphysis = min(component_info, key=lambda item: (abs(item["distance"]), -item["volume"]))
    symphysis["label"] = "PubicSymphysis"

    # Identify left/right by distance from plane
    remaining = [info for info in component_info if info is not symphysis]
    remaining.sort(key=lambda item: item["distance"])
    left = remaining[0]
    right = remaining[1]

    # Ensure 'left' is the negative side, 'right' positive
    if left["distance"] > right["distance"]:
        left, right = right, left
    if left["distance"] > 0 and right["distance"] < 0:
        left, right = right, left

    left["label"] = "SIJCartilageLeft"
    right["label"] = "SIJCartilageRight"

    # Assign region IDs
    label_to_id = {
        "SIJCartilageLeft": 1,
        "SIJCartilageRight": 2,
        "PubicSymphysis": 3,
    }
    for label, rid in label_to_id.items():
        region_names[rid] = label

    for info in component_info:
        label = info["label"]
        rid = label_to_id[label]
        material_map[info["cells"]] = rid

    return material_map, region_names


def mirror_name(name: str) -> str:
    """Swap 'left' <-> 'right' in a name, preserving case.

    Handles both tokenized ("..._left_...") and embedded ("...Left...") forms.
    """

    def repl_left(m: re.Match[str]) -> str:
        s = m.group(0)
        if s.isupper():
            return "RIGHT"
        if s.islower():
            return "right"
        if s[0].isupper():
            return "Right"
        return "right"

    def repl_right(m: re.Match[str]) -> str:
        s = m.group(0)
        if s.isupper():
            return "LEFT"
        if s.islower():
            return "left"
        if s[0].isupper():
            return "Left"
        return "left"

    if re.search(r"left", name, flags=re.IGNORECASE):
        return re.sub(r"left", repl_left, name, flags=re.IGNORECASE)
    if re.search(r"right", name, flags=re.IGNORECASE):
        return re.sub(r"right", repl_right, name, flags=re.IGNORECASE)
    return name


def mirror_point(point: np.ndarray, normal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Reflect point across symmetry plane."""
    n_hat = normal / np.linalg.norm(normal)
    return point - 2.0 * np.dot(point - origin, n_hat) * n_hat


def extract_boundary_faces(conn: np.ndarray) -> set[frozenset[int]]:
    """Extract boundary faces (triangles appearing only once) from tetrahedral mesh."""
    faces = []
    for a, b, c, d in conn:
        faces.extend(
            [
                tuple(sorted((a + 1, b + 1, c + 1))),
                tuple(sorted((a + 1, b + 1, d + 1))),
                tuple(sorted((a + 1, c + 1, d + 1))),
                tuple(sorted((b + 1, c + 1, d + 1))),
            ]
        )
    counts = Counter(faces)
    return {frozenset(tri) for tri, cnt in counts.items() if cnt == 1}


def remap_discrete_sets(
    orig_sets: dict[str, np.ndarray],
    orig_points: np.ndarray,
    tree: KDTree,
) -> dict[str, np.ndarray]:
    """Remap discrete set node indices using KDTree."""
    mapped: dict[str, np.ndarray] = {}
    for set_name, pairs in orig_sets.items():
        remapped = []
        for a, b in pairs:
            p1 = orig_points[a - 1]
            p2 = orig_points[b - 1]
            i1 = int(tree.query(p1, k=1)[1]) + 1
            i2 = int(tree.query(p2, k=1)[1]) + 1
            remapped.append((i1, i2))
        mapped[set_name] = np.asarray(remapped, dtype=int)
    return mapped


def mirror_discrete_sets(
    orig_sets: dict[str, np.ndarray],
    orig_points: np.ndarray,
    tree: KDTree,
    normal: np.ndarray,
    origin: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create mirrored discrete sets for left/right ligaments."""
    mirrored: dict[str, np.ndarray] = {}
    for set_name, pairs in orig_sets.items():
        if "left" not in set_name.lower() and "right" not in set_name.lower():
            continue
        mirror = mirror_name(set_name)
        if mirror in orig_sets:
            continue
        remapped = []
        for a, b in pairs:
            p1 = orig_points[a - 1]
            p2 = orig_points[b - 1]
            p1m = mirror_point(p1, normal, origin)
            p2m = mirror_point(p2, normal, origin)
            i1 = int(tree.query(p1m, k=1)[1]) + 1
            i2 = int(tree.query(p2m, k=1)[1]) + 1
            remapped.append((i1, i2))
        mirrored[mirror] = np.asarray(remapped, dtype=int)
    return mirrored


def remap_surfaces(
    orig_surfaces: dict[str, np.ndarray],
    orig_points: np.ndarray,
    tree: KDTree,
    boundary_faces: set[frozenset[int]],
) -> dict[str, np.ndarray]:
    """Remap surface triangles using KDTree, keeping only boundary faces."""
    mapped: dict[str, np.ndarray] = {}
    for surf_name, tris in orig_surfaces.items():
        valid = []
        for i0, j0, k0 in tris:
            i = int(tree.query(orig_points[i0], k=1)[1]) + 1
            j = int(tree.query(orig_points[j0], k=1)[1]) + 1
            k = int(tree.query(orig_points[k0], k=1)[1]) + 1
            if frozenset((i, j, k)) in boundary_faces:
                valid.append((i, j, k))
        if valid:
            mapped[surf_name] = np.asarray(valid, dtype=int)
    return mapped


def mirror_surfaces(
    orig_surfaces: dict[str, np.ndarray],
    orig_points: np.ndarray,
    tree: KDTree,
    normal: np.ndarray,
    origin: np.ndarray,
    boundary_faces: set[frozenset[int]],
) -> dict[str, np.ndarray]:
    """Create mirrored surfaces for left/right anatomy."""
    mirrored: dict[str, np.ndarray] = {}
    for surf_name, tris in orig_surfaces.items():
        if "left" not in surf_name.lower() and "right" not in surf_name.lower():
            continue
        mirror = mirror_name(surf_name)
        if mirror in orig_surfaces:
            continue
        valid = []
        for i0, j0, k0 in tris:
            pi = orig_points[i0]
            pj = orig_points[j0]
            pk = orig_points[k0]
            pim = mirror_point(pi, normal, origin)
            pjm = mirror_point(pj, normal, origin)
            pkm = mirror_point(pk, normal, origin)
            im = int(tree.query(pim, k=1)[1]) + 1
            jm = int(tree.query(pjm, k=1)[1]) + 1
            km = int(tree.query(pkm, k=1)[1]) + 1
            if frozenset((im, jm, km)) in boundary_faces:
                valid.append((im, jm, km))
        if valid:
            mirrored[mirror] = np.asarray(valid, dtype=int)
    return mirrored


def main() -> None:
    """Build FEBio model with material regions and mirrored ligaments."""
    cfg = SimulationConfig()
    template = pv.read("data/pelvic.vtk")

    # Parse initial FEBio model
    f2x = FEBio2Dolfinx("anatomy_data/initial_model.feb")
    normal, origin = find_symmetry_plane(f2x.mesh_dolfinx.geometry.x)

    # Extract mesh
    topo, cell_types, coords = dplot.vtk_mesh(f2x.mesh_dolfinx, f2x.mesh_dolfinx.topology.dim)
    grid = pv.UnstructuredGrid(topo, cell_types, coords)
    conn = grid.cells.reshape(-1, 5)[:, 1:5]
    coords_arr = np.asarray(coords, dtype=float)
    span = float(np.linalg.norm(coords_arr.max(axis=0) - coords_arr.min(axis=0)))

    # Identify material regions
    material_map, region_names = identify_material_regions(
        grid, conn, template, normal, origin, float(cfg["REGION_TOLERANCE"]), span
    )
    print("Regions:", {rid: region_names[rid] for rid in sorted(region_names)})

    # Build KDTree for remapping
    feb_points = np.vstack(list(f2x.nodes.values()))
    tree = KDTree(coords_arr)

    # Remap discrete sets and create mirrors
    mapped_sets = remap_discrete_sets(f2x.discrete_sets, feb_points, tree)
    mirrored_sets = mirror_discrete_sets(f2x.discrete_sets, feb_points, tree, normal, origin)
    mapped_sets.update(mirrored_sets)

    # Extract boundary faces
    boundary_faces = extract_boundary_faces(conn)

    # Remap surfaces and create mirrors
    mapped_surfaces = remap_surfaces(f2x.surfaces, feb_points, tree, boundary_faces)
    mirrored_surfaces = mirror_surfaces(
        f2x.surfaces, feb_points, tree, normal, origin, boundary_faces
    )
    mapped_surfaces.update(mirrored_surfaces)

    # Material properties
    region_materials: dict[str, dict[str, float]] = {}
    for rid in np.unique(material_map):
        label = region_names[int(rid)]
        if rid == 0:
            region_materials[label] = {"E": 10000.0, "v": cfg["BONE_POISSON_RATIO"]}
        elif "symph" in label.lower():
            region_materials[label] = {
                "E": cfg["SYMPHYSIS_MODULUS"],
                "v": cfg["SYMPHYSIS_POISSON_RATIO"],
            }
        else:
            region_materials[label] = {
                "E": cfg["SIJ_CARTILAGE_MODULUS"],
                "v": cfg["SIJ_CARTILAGE_POISSON_RATIO"],
            }

    # Write output
    out_path = Path("anatomy_data/full_model.feb")
    write_febio_xml(
        out_path,
        coords_arr,
        conn,
        material_map,
        region_names,
        region_materials,
        mapped_sets,
        mapped_surfaces,
    )
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
