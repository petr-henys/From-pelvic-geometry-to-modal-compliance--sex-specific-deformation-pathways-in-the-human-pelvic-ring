"""Map template-driven density and shape data onto the FEBio mesh."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from scipy.interpolate import RBFInterpolator
from dolfinx import plot


if TYPE_CHECKING:  # pragma: no cover - typing only
    from dolfinx import fem
    from simulation.febio_parser import FEBio2Dolfinx

__all__ = ["calculate_bone_modulus", "ShapeMapper", "MaterialMapper"]

logger = logging.getLogger(__name__)


def calculate_bone_modulus(
    density_field: np.ndarray,
    alpha: float = 10200.0,
    beta: float = 2.0,
    threshold: float = 0.486,
) -> np.ndarray:
    """Convert hydroxyapatite density to elastic modulus (Keller, 1994).
    
    Args:
        density_field: Hydroxyapatite density values
        alpha: Alpha coefficient for high-branch law (default: 10200.0)
        beta: Beta exponent for high-branch law (default: 2.0)
        threshold: Ash density threshold for piecewise law (default: 0.486)
    
    Returns:
        Young's modulus in MPa
    """
    density = np.asarray(density_field, dtype=float)
    density = np.maximum(density, 0.0)
    rho_ash = (density + 0.09) / 1.14
    return np.where(rho_ash <= threshold, 2398.0, alpha * rho_ash**beta)


class ShapeMapper:
    """Interpolate dataset shape fields (φ) from the template points to the FEBio mesh."""

    def __init__(self, phi: "fem.Function", template: pv.PolyData, shapes: np.ndarray, config: dict[str, Any]):
        self.template = template
        self.shapes = np.asarray(shapes)
        self.phi = phi
        self.dof_coordinates = phi.function_space.tabulate_dof_coordinates()
        self.rbf_smoothing = float(config["RBF_DATA_SMOOTHING"])
        self.rbf_neighbors = int(config["RBF_DATA_NEIGHBORS"])

    def apply_reference(self) -> None:
        self.phi.x.array[:] = 0.0
        self.phi.x.scatter_forward()

    def apply_sample(self, idx: int) -> None:
        rbf = RBFInterpolator(
            self.template.points, 
            self.shapes[idx] - self.template.points,
            smoothing=self.rbf_smoothing,
            neighbors=self.rbf_neighbors,
        )
        values = np.asarray(rbf(self.dof_coordinates), dtype=float).ravel()
        self.phi.x.array[:] = values
        self.phi.x.scatter_forward()

    def save_to_vtk(self, filename: str) -> None:
        mesh = self.phi.function_space.mesh
        mesh_topology, cell_types, geometry = plot.vtk_mesh(mesh, mesh.topology.dim)
        grid = pv.UnstructuredGrid(mesh_topology, cell_types, geometry)
        grid.point_data["shape_field"] = self.phi.x.array.reshape((-1, mesh.geometry.dim))
        grid.save(filename)


class MaterialMapper:
    """Map density-driven elasticity fields onto the FEBio mesh."""
    def __init__(
        self,
        E_func: "fem.Function",
        nu_func: "fem.Function",
        template: pv.PolyData,
        densities: np.ndarray,
        config: dict[str, Any],
        febio_parser: "FEBio2Dolfinx",
    ):
        self.template = template
        self.densities = np.asarray(densities)
        self.config = config
        self.E_func = E_func
        self.nu_func = nu_func
        self.material_labels = np.asarray(febio_parser.material_labels, dtype=str)
        self.rbf_smoothing = float(config["RBF_DATA_SMOOTHING"])
        self.rbf_neighbors = int(config["RBF_DATA_NEIGHBORS"])
        self.bone_mask, self.cartilage_mask, self.symphysis_mask = self._material_masks()
        self.dof_coordinates = E_func.function_space.tabulate_dof_coordinates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def num_samples(self) -> int:
        """Return number of samples in dataset."""
        return len(self.densities)

    def apply_reference(self) -> None:
        """Map reference (dataset-mean) elasticity onto FEBio mesh."""
        modulus = self._bone_modulus(self._reference_density())
        rbf = RBFInterpolator(
            self.template.points, 
            modulus, 
            smoothing=self.rbf_smoothing,
            neighbors=self.rbf_neighbors,
        )
        # Bone: interpolated modulus from density
        self.E_func.x.array[:] = np.asarray(rbf(self.dof_coordinates), dtype=float).ravel()
        self.nu_func.x.array[:] = float(self.config["BONE_POISSON_RATIO"])
        self.update_fields()

    def apply_sample(self, idx: int) -> None:
        """Map sample-specific elasticity onto FEBio mesh."""
        modulus = self._bone_modulus(self.densities[idx])
        rbf = RBFInterpolator(
            self.template.points, 
            modulus, 
            smoothing=self.rbf_smoothing,
            neighbors=self.rbf_neighbors,
        )
        # Bone: interpolated modulus from density
        self.E_func.x.array[:] = np.asarray(rbf(self.dof_coordinates), dtype=float).ravel()
        self.nu_func.x.array[:] = float(self.config["BONE_POISSON_RATIO"])
        self.update_fields()

    def compute_bone_param_sensitivity_weights(self, *, sample_idx: int | None = None) -> dict[str, np.ndarray]:
        """Return (dE_cell/dparameter)/E_cell for the implemented bone map.

        The source-point modulus is evaluated before RBF interpolation. Its
        parameter derivatives must be interpolated by the same RBF operator;
        interpolating density and differentiating afterwards is not equivalent.
        Non-bone cells have fixed moduli and therefore zero derivative.
        """
        alpha = float(self.config["BONE_MODULUS_ALPHA"])
        beta = float(self.config["BONE_MODULUS_BETA"])
        threshold = float(self.config["BONE_MODULUS_THRESHOLD"])
        density = self._reference_density() if sample_idx is None else np.asarray(self.densities[int(sample_idx)])
        rho_ash = (np.maximum(density, 0.0) + 0.09) / 1.14
        active_source = rho_ash > threshold
        high = rho_ash**beta
        dE_dalpha_source = np.where(active_source, high, 0.0)
        dE_dbeta_source = np.where(active_source, alpha * high * np.log(rho_ash), 0.0)
        derivatives = {}
        for name, source in (("alpha", dE_dalpha_source), ("beta", dE_dbeta_source)):
            rbf = RBFInterpolator(
                self.template.points, source,
                smoothing=self.rbf_smoothing, neighbors=self.rbf_neighbors,
            )
            dE = np.asarray(rbf(self.dof_coordinates), dtype=float).ravel()
            weights = np.zeros_like(dE)
            E = self.E_func.x.array
            valid = self.bone_mask & np.isfinite(E) & (np.abs(E) > 1e-12)
            weights[valid] = dE[valid] / E[valid]
            derivatives[name] = weights
        return derivatives

    def save_to_vtk(self, filename: str) -> None:
        mesh_topology, cell_types, geometry = plot.vtk_mesh(
            self.E_func.function_space.mesh, self.E_func.function_space.mesh.topology.dim
        )
        grid = pv.UnstructuredGrid(mesh_topology, cell_types, geometry)
        grid.cell_data["E"] = np.copy(self.E_func.x.array)
        grid.cell_data["nu"] = np.copy(self.nu_func.x.array)
        grid.save(filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _material_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels_lower = np.char.lower(self.material_labels)
        contains = lambda needle: np.char.find(labels_lower, needle) >= 0

        bone_mask = (
            contains("bone")
            & (np.char.find(labels_lower, "cartilage") < 0)
            & (np.char.find(labels_lower, "symph") < 0)
        )
        cartilage_mask = contains("cartilage")
        symphysis_mask = contains("symph")

        if not bone_mask.any():
            logger.warning("No bone regions identified in FEBio domains; defaulting all cells to bone.")
            bone_mask = np.ones(labels_lower.size, dtype=bool)
        if not cartilage_mask.any():
            logger.info("No cartilage regions found in FEBio domains.")
        if not symphysis_mask.any():
            logger.info("No symphysis regions found in FEBio domains.")

        return bone_mask, cartilage_mask, symphysis_mask

    def _reference_density(self) -> np.ndarray:
        return np.median(self.densities, axis=0)

    def _bone_modulus(self, density: np.ndarray) -> np.ndarray:
        alpha = float(self.config["BONE_MODULUS_ALPHA"])
        beta = float(self.config["BONE_MODULUS_BETA"])
        threshold = float(self.config["BONE_MODULUS_THRESHOLD"])
        return calculate_bone_modulus(density, alpha=alpha, beta=beta, threshold=threshold)

    def update_fields(self) -> None:

        # Cartilage regions
        self.E_func.x.array[self.cartilage_mask] = float(self.config["SIJ_CARTILAGE_MODULUS"])
        self.nu_func.x.array[self.cartilage_mask] = float(self.config["SIJ_CARTILAGE_POISSON_RATIO"])

        # Symphysis regions
        self.E_func.x.array[self.symphysis_mask] = float(self.config["SYMPHYSIS_MODULUS"])
        self.nu_func.x.array[self.symphysis_mask] = float(self.config["SYMPHYSIS_POISSON_RATIO"])

        self.E_func.x.scatter_forward()
        self.nu_func.x.scatter_forward()
