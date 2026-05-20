from __future__ import annotations

import logging
import numpy as np
import pyvista as pv
from petsc4py import PETSc
from dolfinx import fem
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator


logger = logging.getLogger(__name__)

class LigamentSpringSystem:
    """Assemble ligament springs into a PETSc stiffness matrix and pretension force vector.

    - Maps each polyline bundle to pairs of FE vector‑DoF blocks via a KDTree built on
      deformed coordinates ``X + φ(X)`` (φ interpolated by an RBF).
    - Stiffness contributions include a material term and an optional geometric term
      from pretension.
    - Single MPI rank is required; pretensions must be non‑negative.
    
    Conventions
    -----------
    - Let ``e`` be the unit direction of a segment (in the deformed configuration) and ``L`` its length.
    - Material term per segment uses ``W = k_seg · (e ⊗ e)`` with no ``1/L`` factor.
      Here, ``k_seg`` is the per‑segment axial spring stiffness [N/m]. The constructor
      accepts per‑bundle stiffness parameters (one per bundle) which are distributed
      equally among the bundle’s segments, i.e. ``k_seg = k_total / n_segments``.
    - Geometric term from pretension ``T`` uses the classical form ``(T/L) · (I − e ⊗ e)`` per segment,
      again with the bundle’s total pretension equally distributed per segment.
    """

    def __init__(
        self,
        V: fem.FunctionSpace,
        ligament_pairs: list[pv.PolyData],
        stiffnesses: list[float],
        phi: fem.Function,
        config: dict,
        bundle_names: list[str] | None = None,
        pretensions: list[float] | None = None,
    ) -> None:
        """Initialize ligament spring system.

        Args:
            V: Vector function space for displacements.
            ligament_pairs: Polyline bundles (one per ligament), each as ``pv.PolyData``.
            stiffnesses: Effective axial spring stiffness parameter per bundle [N/m];
                distributed equally across segments when assembling the material term.
            phi: Geometry mapping field in the same vector space (used to build the KDTree on
                deformed coordinates via an RBF interpolator).
            config: Configuration dictionary (expects RBF settings and mapping tolerance).
            bundle_names: Optional names for bundles; used for labeling/metadata.
            pretensions: Optional pretension forces per bundle [N], must be ≥ 0; equally
                distributed per segment for the geometric stiffness and pretension RHS.
        """
        self.V = V
        self.comm = V.mesh.comm
        assert self.comm.size == 1, "Single MPI rank required"
        assert len(ligament_pairs) == len(stiffnesses)
        assert bundle_names is None or len(bundle_names) == len(ligament_pairs)
        assert pretensions is None or len(pretensions) == len(ligament_pairs)

        self._phi = phi
        self.config = config
        self._gdim = V.mesh.geometry.dim
        self.bs = V.dofmap.index_map_bs
        self._X = V.tabulate_dof_coordinates()
        self.num_blocks = self._X.shape[0]
        assert self.bs == self._gdim, "Block size must equal geometry dimension"

        self._fu = self._build_fu_interpolator()
        self._deps = float(config["LIGAMENT_MAPPING_TOLERANCE"])
        self._stiffnesses = list(stiffnesses)
        self._bundle_names = (
            [f"bundle_{i}" for i in range(len(ligament_pairs))]
            if bundle_names is None
            else list(bundle_names)
        )
        self._pretension_total = (
            [0.0] * len(ligament_pairs)
            if pretensions is None
            else list(pretensions)
        )
        assert all(p >= 0.0 for p in self._pretension_total), "Pretensions must be ≥0"

        self._pairs_per_bundle: list[set[tuple[int, int]]] = []
        # Per-bundle diagnostics from mapping stage
        self._bundle_map_stats: list[dict] = []
        # Segment geometry cache: (inv_nseg, [(i, j, e), ...])
        self._segment_geometry: list[tuple[float, list[tuple[int, int, np.ndarray]]]] = []
        # Per-bundle diagnostics from assembly stage
        self._bundle_assembly_stats: list[dict] = []

        self._build_mapper()
        self._map_geometries_to_block_dofs(ligament_pairs)
        self._create_global_K()
        self._fill_global_K()
        self._assemble_pretension_vector()

        logger.debug("LigamentSpringSystem initialized: %d bundles, %d blocks", len(ligament_pairs), self.num_blocks)

    def _build_fu_interpolator(self) -> RBFInterpolator:
        """Build RBF interpolator from current phi values."""
        X = self.V.tabulate_dof_coordinates()
        gdim = self.V.mesh.geometry.dim
        phi_vals = self._phi.x.array.reshape((-1, gdim))
        smoothing = float(self.config["RBF_SOLVER_SMOOTHING"])
        neighbors = self.config["RBF_SOLVER_NEIGHBORS"]
        return RBFInterpolator(X, phi_vals, neighbors=neighbors, smoothing=smoothing)

    def _build_mapper(self) -> None:
        """Build KDTree on deformed coordinates to map polyline points to FE DoF blocks."""
        # Ligament polyline points are in physical/sample coordinates
        # KDTree must be built on deformed coordinates: X + phi(X)
        self._Xd = self._X + self._fu(self._X)
        
        self._kdt = KDTree(self._Xd)

    @staticmethod
    def _poly_segments(poly: pv.PolyData) -> list[tuple[int, int]]:
        """Extract consecutive point pairs from PyVista polyline representation.
        
        Parses poly.lines array format: [npts, id0, id1, ..., npts, id0, id1, ...]
        """
        lines = np.asarray(poly.lines, dtype=np.int32)
        segs: list[tuple[int, int]] = []
        i = 0
        n = lines.size
        while i < n:
            npts = int(lines[i])
            idxs = list(map(int, lines[i + 1 : i + 1 + npts]))
            segs.extend(zip(idxs[:-1], idxs[1:]))
            i += 1 + npts
        return segs

    def _map_geometries_to_block_dofs(self, ligament_pairs: list[pv.PolyData]) -> None:
        """Map polyline points to nearest vector‑DoF block indices within tolerance (deformed config)."""
        self._pairs_per_bundle.clear()
        self._bundle_map_stats.clear()
        for b_idx, poly in enumerate(ligament_pairs):
            pts = np.asarray(poly.points, dtype=float)
            n_pts = int(pts.shape[0])
            # Query against deformed coordinates KDTree
            dists, idxs = self._kdt.query(pts)
            mask = dists <= self._deps
            n_mapped = int(np.count_nonzero(mask))
            mean_dist = float(dists[mask].mean()) if n_mapped else float("nan")
            max_dist = float(dists[mask].max()) if n_mapped else float("nan")
            min_dist = float(dists[mask].min()) if n_mapped else float("nan")

            idxs = np.where(mask, idxs, -1)
            undirected: set[tuple[int, int]] = set()

            raw_segments = self._poly_segments(poly)
            n_pairs_total = len(raw_segments)
            n_pairs_valid_raw = 0
            for a, b in raw_segments:
                ia = int(idxs[a])
                ib = int(idxs[b])
                if ia < 0 or ib < 0 or ia == ib:
                    continue
                n_pairs_valid_raw += 1
                if ia < ib:
                    undirected.add((ia, ib))
                else:
                    undirected.add((ib, ia))

            n_pairs_kept = len(undirected)
            n_pairs_invalid = n_pairs_total - n_pairs_valid_raw
            n_pairs_dups_removed = n_pairs_valid_raw - n_pairs_kept

            map_stats = {
                "bundle": self._bundle_names[b_idx] if b_idx < len(self._bundle_names) else f"bundle_{b_idx}",
                "n_poly_points": n_pts,
                "n_mapped_points": n_mapped,
                "mapped_pct": (100.0 * n_mapped / n_pts) if n_pts else 0.0,
                "nn_dist_mean": mean_dist,
                "nn_dist_min": min_dist,
                "nn_dist_max": max_dist,
                "tol": self._deps,
                "n_pairs_total": n_pairs_total,
                "n_pairs_valid_raw": n_pairs_valid_raw,
                "n_pairs_kept": n_pairs_kept,
                "n_pairs_invalid": n_pairs_invalid,
                "n_pairs_dups_removed": n_pairs_dups_removed,
            }
            self._bundle_map_stats.append(map_stats)

            logger.debug(
                ("[Ligament map] bundle=%s pts=%d mapped=%d (%.1f%%) nn_dist(mean/min/max)=(%.4g/%.4g/%.4g) tol=%.4g | "
                 "segs total=%d valid_raw=%d kept=%d invalid=%d dups_removed=%d"),
                map_stats["bundle"], n_pts, n_mapped, map_stats["mapped_pct"],
                mean_dist, min_dist, max_dist, self._deps,
                n_pairs_total, n_pairs_valid_raw, n_pairs_kept, n_pairs_invalid, n_pairs_dups_removed
            )

            self._pairs_per_bundle.append(undirected)

    def _create_global_K(self) -> None:
        """Create an empty PETSc AIJ matrix with vector block size for ligament K."""
        N = self.num_blocks * self.bs
        self.K = PETSc.Mat().createAIJ(size=(N, N), comm=self.comm)
        self.K.setType(PETSc.Mat.Type.AIJ)
        self.K.setBlockSize(self.bs)
        self.K.setUp()

    def _refresh_mapper(self) -> None:
        """Refresh deformed coordinates and zero K entries (preserve sparsity pattern)."""
        self._Xd = self._X + self._fu(self._X)
        self.K.zeroEntries()

    def _fill_global_K(self) -> None:
        """Assemble ligament stiffness matrix entries (material + geometric terms)."""
        self._refresh_mapper()
        self._segment_geometry = []
        self._bundle_assembly_stats = []

        Xd = self._Xd
        total_springs = 0
        total_used = 0
        total_skipped = 0
        for bundle_idx, (bundle_pairs, k_total) in enumerate(zip(self._pairs_per_bundle, self._stiffnesses)):
            nseg = len(bundle_pairs)
            inv_nseg = 1.0 / float(nseg) if nseg else 0.0
            bundle_segment_data: list[tuple[int, int, np.ndarray]] = []
            if nseg == 0:
                self._segment_geometry.append((inv_nseg, bundle_segment_data))
                self._bundle_assembly_stats.append({
                    "bundle": self._bundle_names[bundle_idx] if bundle_idx < len(self._bundle_names) else f"bundle_{bundle_idx}",
                    "n_seg": 0, "n_used": 0, "n_skipped": 0, "L_total": 0.0,
                    "L_mean": float("nan"), "L_min": float("nan"), "L_max": float("nan"),
                    "k_total": float(k_total), "EA_per": 0.0,
                    "T_total": float(self._pretension_total[bundle_idx]) if bundle_idx < len(self._pretension_total) else 0.0,
                    "T_per": 0.0
                })
                continue

            EA_per = k_total / float(nseg) if k_total != 0.0 else 0.0
            T_total = self._pretension_total[bundle_idx] if bundle_idx < len(self._pretension_total) else 0.0
            T_per = T_total / float(nseg) if T_total != 0.0 else 0.0
            I = np.eye(self.bs, dtype=float)

            if k_total == 0.0:
                # Material contribution is zero; warn once per bundle.
                logger.warning("Zero stiffness for bundle %d, skipping material contributions", bundle_idx)

            L_list = []
            used = 0
            skipped = 0

            for i, j in bundle_pairs:
                xi = Xd[i]
                xj = Xd[j]
                d = xj - xi
                L = float(np.linalg.norm(d))
                if not np.isfinite(L) or L <= self._deps:
                    skipped += 1
                    continue
                e = d / L
                L_list.append(L)
                bundle_segment_data.append((i, j, e.copy()))
                used += 1

                # Material stiffness
                W = EA_per * np.outer(e, e)
                # Geometric stiffness from pretension
                Wg = (T_per / L) * (I - np.outer(e, e)) if T_per != 0.0 else None

                rows_i = list(range(i * self.bs, (i + 1) * self.bs))
                rows_j = list(range(j * self.bs, (j + 1) * self.bs))
                cols_i = rows_i
                cols_j = rows_j

                self.K.setValues(rows_i, cols_i, W, addv=PETSc.InsertMode.ADD_VALUES)
                self.K.setValues(rows_i, cols_j, -W, addv=PETSc.InsertMode.ADD_VALUES)
                self.K.setValues(rows_j, cols_i, -W, addv=PETSc.InsertMode.ADD_VALUES)
                self.K.setValues(rows_j, cols_j, W, addv=PETSc.InsertMode.ADD_VALUES)

                if Wg is not None:
                    self.K.setValues(rows_i, cols_i, Wg, addv=PETSc.InsertMode.ADD_VALUES)
                    self.K.setValues(rows_i, cols_j, -Wg, addv=PETSc.InsertMode.ADD_VALUES)
                    self.K.setValues(rows_j, cols_i, -Wg, addv=PETSc.InsertMode.ADD_VALUES)
                    self.K.setValues(rows_j, cols_j, Wg, addv=PETSc.InsertMode.ADD_VALUES)

            self._segment_geometry.append((inv_nseg, bundle_segment_data))

            L_total = float(np.sum(L_list)) if L_list else 0.0
            L_mean = float(np.mean(L_list)) if L_list else float("nan")
            L_min = float(np.min(L_list)) if L_list else float("nan")
            L_max = float(np.max(L_list)) if L_list else float("nan")

            asm_stats = {
                "bundle": self._bundle_names[bundle_idx] if bundle_idx < len(self._bundle_names) else f"bundle_{bundle_idx}",
                "n_seg": int(nseg),
                "n_used": int(used),
                "n_skipped": int(skipped),
                "L_total": L_total,
                "L_mean": L_mean,
                "L_min": L_min,
                "L_max": L_max,
                "k_total": float(k_total),
                "EA_per": float(EA_per),
                "T_total": float(T_total),
                "T_per": float(T_per),
            }
            self._bundle_assembly_stats.append(asm_stats)

            logger.debug(
                ("[Ligament asm] bundle=%s n_seg=%d used=%d skipped=%d | "
                 "L_total=%.6g L_mean=%.6g L_min=%.6g L_max=%.6g | "
                 "k_total=%.6g EA_per=%.6g T_total=%.6g T_per=%.6g"),
                asm_stats["bundle"], asm_stats["n_seg"], asm_stats["n_used"], asm_stats["n_skipped"],
                L_total, L_mean, L_min, L_max,
                asm_stats["k_total"], asm_stats["EA_per"], asm_stats["T_total"], asm_stats["T_per"]
            )

            total_springs += nseg
            total_used += used
            total_skipped += skipped

        self.K.assemble()
        logger.debug(
            "Ligament K assembled (bundles=%d springs=%d used=%d skipped=%d)",
            len(self._pairs_per_bundle), total_springs, total_used, total_skipped,
        )

    def _assemble_pretension_vector(self) -> None:
        """Assemble RHS force vector due to ligament pretensions.

        Distributes each bundle’s total pretension equally across its segments,
        applying ±T_per · e forces at segment endpoint blocks (i, j) consistent
        with the geometric stiffness sign convention.
        """
        N = self.num_blocks * self.bs
        self.fpret = PETSc.Vec().createMPI(N, comm=self.comm)
        self.fpret.set(0.0)
        if not self._segment_geometry:
            self.fpret.assemble()
            return

        for (segments_info, T_total) in zip(self._segment_geometry, self._pretension_total):
            inv_nseg, segments = segments_info
            if not segments or T_total == 0.0:
                continue
            T_per = float(T_total) * inv_nseg
            for i, j, e in segments:
                rows_i = list(range(i * self.bs, (i + 1) * self.bs))
                rows_j = list(range(j * self.bs, (j + 1) * self.bs))
                self.fpret.setValues(rows_i, T_per * e, addv=PETSc.InsertMode.ADD_VALUES)
                self.fpret.setValues(rows_j, -T_per * e, addv=PETSc.InsertMode.ADD_VALUES)
        self.fpret.assemble()

    def update(self, stiffnesses: list[float] | None = None, pretensions: list[float] | None = None) -> PETSc.Mat:
        """Reassemble values (same sparsity) using current φ/stiffnesses/pretensions.

        Args:
            stiffnesses: Optional new bundle stiffness parameters [N/m]. If ``None``, keep current values.
            pretensions: Optional new bundle pretensions [N]. If ``None``, keep current values.
        """
        if stiffnesses is not None:
            if len(stiffnesses) != len(self._stiffnesses):
                raise ValueError(f"stiffnesses must have length {len(self._stiffnesses)}, got {len(stiffnesses)}")
            self._stiffnesses = list(stiffnesses)
        
        if pretensions is not None:
            if len(pretensions) != len(self._pretension_total):
                raise ValueError(f"pretensions must have length {len(self._pretension_total)}, got {len(pretensions)}")
            self._pretension_total = [float(p) for p in pretensions]
            if any(p < 0.0 for p in self._pretension_total):
                raise ValueError("pretensions must be non-negative")
        
        self._fu = self._build_fu_interpolator()
        self._fill_global_K()
        self._assemble_pretension_vector()
        return self.K

    @property
    def num_bundles(self) -> int:
        return len(self._stiffnesses)

    def mode_bundle_quadratic_forms(self, mode_vectors: np.ndarray) -> np.ndarray:
        """Return quadratic forms ``xᵀ (∂K/∂k_i) x`` per mode and bundle.

        The derivative is with respect to the bundle stiffness parameter (per-segment
        material contribution). No slack/activation gating is applied here. This matches
        the material tangent contribution k (e⊗e) accumulated over bundle segments and
        averaged by the number of segments.
        """
        num_bundles = len(self._segment_geometry)
        if mode_vectors.ndim == 2:
            mode_vectors = mode_vectors[np.newaxis, ...]
        num_modes = mode_vectors.shape[0]
        result = np.zeros((num_modes, num_bundles), dtype=float)
        if num_bundles == 0 or num_modes == 0:
            return result

        for mode_idx, vec in enumerate(mode_vectors):
            blocks = np.ascontiguousarray(vec, dtype=float)
            if blocks.shape[0] != self.num_blocks or blocks.shape[1] != self.bs:
                raise ValueError(
                    f"Mode vector shape {blocks.shape} incompatible with ligament system "
                    f"blocks=({self.num_blocks}, {self.bs})"
                )
            for bundle_idx, (inv_nseg, segments) in enumerate(self._segment_geometry):
                if inv_nseg == 0.0 or not segments:
                    continue
                accum = 0.0
                for i, j, direction in segments:
                    # axial projection of the relative displacement of the segment end nodes
                    du = blocks[j] - blocks[i]
                    dL = float(np.dot(du, direction))
                    accum += dL * dL
                result[mode_idx, bundle_idx] = accum * inv_nseg
        return result


    def mode_bundle_geometric_quadratic_forms(self, mode_vectors: np.ndarray) -> np.ndarray:
        """Return quadratic forms ``xᵀ (∂K/∂T_i) x`` for pretension sensitivities per bundle.

        Uses the geometric stiffness contribution per segment ``(1/nseg)·(1/L)·(I − e ⊗ e)``;
        the derivative is independent of the current pretension magnitude.
        """
        if mode_vectors.ndim == 2:
            mode_vectors = mode_vectors[np.newaxis, ...]
        num_modes = int(mode_vectors.shape[0])
        num_bundles = len(self._segment_geometry)
        out = np.zeros((num_modes, num_bundles), dtype=float)
        if num_modes == 0 or num_bundles == 0:
            return out

        Xd = self._Xd

        for m in range(num_modes):
            blocks = np.ascontiguousarray(mode_vectors[m], dtype=float)
            if blocks.shape != (self.num_blocks, self.bs):
                raise ValueError(
                    f"Mode vector shape {blocks.shape} incompatible with ligament system blocks=({self.num_blocks}, {self.bs})"
                )
            for b, (inv_nseg, segments) in enumerate(self._segment_geometry):
                if inv_nseg == 0.0 or not segments:
                    continue
                accum = 0.0
                for i, j, e in segments:
                    d = Xd[j] - Xd[i]
                    L = float(np.linalg.norm(d))
                    if not np.isfinite(L) or L <= self._deps:
                        continue
                    du = blocks[j] - blocks[i]
                    t = float(np.dot(du, e))
                    transverse_sq = float(np.dot(du, du)) - t * t
                    accum += transverse_sq / L
                out[m, b] = accum * inv_nseg
        return out

    @property
    def bundle_names(self) -> list[str]:
        """Return names (labels) of ligament bundles in the assembled order."""
        return list(self._bundle_names)

    @property
    def pretension_vector(self) -> PETSc.Vec:
        """Global RHS vector of pretension forces (size matches global DOFs)."""
        return self.fpret

    @property
    def map_debug_stats(self) -> list[dict]:
        """Diagnostics collected during mapping from polylines to FE blocks (per bundle)."""
        return list(self._bundle_map_stats)

    @property
    def assembly_debug_stats(self) -> list[dict]:
        """Diagnostics collected during stiffness assembly (per bundle): n_seg, used/skipped, length stats, etc."""
        return list(self._bundle_assembly_stats)