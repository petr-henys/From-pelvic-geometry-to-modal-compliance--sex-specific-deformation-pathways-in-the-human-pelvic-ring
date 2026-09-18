"""Centralized publication-quality 3D rendering module using PyVista / VTK.

Provides standardized studio lighting, anti-aliasing (SSAA), depth peeling for
order-independent transparency, SSAO for solid anatomical specimens, camera
conventions, and high-resolution deterministic output across all manuscript figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pyvista as pv

logger = logging.getLogger("publication_rendering")


def log_rendering_environment() -> dict[str, str]:
    """Inspect and log rendering backend capabilities (GPU, OpenGL, PyVista, VTK)."""
    info = {
        "pyvista_version": str(pv.__version__),
    }
    try:
        import vtk
        info["vtk_version"] = str(vtk.vtkVersion.GetVTKVersion())
    except Exception as exc:
        info["vtk_version"] = f"unknown ({exc})"

    try:
        gpu_info = pv.GPUInfo()
        info["gpu_renderer"] = str(gpu_info.renderer)
        info["gpu_version"] = str(gpu_info.version)
        info["gpu_vendor"] = str(gpu_info.vendor)
    except Exception as exc:
        info["gpu_info"] = f"unavailable ({exc})"

    logger.info("3D Rendering Environment: %s", info)
    return info


def configure_studio_lighting(
    plotter: pv.Plotter,
    key_intensity: float = 0.75,
    fill_intensity: float = 0.35,
    rim_intensity: float = 0.20,
    key_pos: tuple[float, float, float] = (1.0, 0.8, 1.2),
    fill_pos: tuple[float, float, float] = (-1.0, -0.6, 0.8),
    rim_pos: tuple[float, float, float] = (0.0, 1.5, -1.0),
    key_color: str = "white",
    fill_color: str = "#f0f4f8",
    rim_color: str = "white",
) -> None:
    """Replace default VTK headlight with a balanced 3-point studio lighting setup.

    Uses camera-relative coordinates ('cameralight') so key, fill, and rim
    highlights remain consistent across all anatomical viewpoints.
    """
    plotter.remove_all_lights()

    # 1. Key light: primary directional modeler (upper-right front)
    key = pv.Light(
        position=key_pos,
        focal_point=(0.0, 0.0, 0.0),
        light_type="cameralight",
        intensity=key_intensity,
        color=key_color,
    )
    # 2. Fill light: softer contrast reducer (lower-left front, cool daylight tint)
    fill = pv.Light(
        position=fill_pos,
        focal_point=(0.0, 0.0, 0.0),
        light_type="cameralight",
        intensity=fill_intensity,
        color=fill_color,
    )
    # 3. Rim / back light: subtle contour separator from neutral background
    rim = pv.Light(
        position=rim_pos,
        focal_point=(0.0, 0.0, 0.0),
        light_type="cameralight",
        intensity=rim_intensity,
        color=rim_color,
    )

    plotter.add_light(key)
    plotter.add_light(fill)
    plotter.add_light(rim)


def create_publication_plotter(
    window_size: tuple[int, int] = (1200, 1000),
    background: str = "white",
    enable_ssaa: bool = True,
    enable_ssao: bool = False,
    ssao_radius: float = 25.0,
    ssao_bias: float = 0.005,
    ssao_kernel_size: int = 256,
    enable_depth_peeling: bool = False,
    depth_peels: int = 8,
) -> pv.Plotter:
    """Create and configure a high-end publication off-screen PyVista plotter.

    Parameters
    ----------
    window_size : tuple[int, int]
        Off-screen framebuffer resolution.
    background : str
        Background color (default 'white' for print).
    enable_ssaa : bool
        Enable Super-Sample Anti-Aliasing (SSAA) for crisp, aliasing-free edges.
    enable_ssao : bool
        Enable Screen Space Ambient Occlusion (ideal for solid anatomy, disable
        when rendering overlapping transparent ghost meshes).
    ssao_radius : float
        SSAO radius calibrated to pelvic physical dimensions (~20-30 mm).
    enable_depth_peeling : bool
        Enable depth peeling for accurate order-independent transparency.
    """
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)

    # Lighting
    configure_studio_lighting(plotter)

    # Anti-aliasing
    if enable_ssaa:
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception as exc:
            logger.warning("SSAA anti-aliasing unavailable, falling back: %s", exc)

    # Ambient Occlusion (for solid geometry)
    if enable_ssao:
        try:
            plotter.enable_ssao(
                radius=ssao_radius,
                bias=ssao_bias,
                kernel_size=ssao_kernel_size,
                blur=True,
            )
        except Exception as exc:
            logger.warning("SSAO unavailable: %s", exc)

    # Depth Peeling (for order-independent transparency)
    if enable_depth_peeling:
        try:
            plotter.enable_depth_peeling(number_of_peels=depth_peels, occlusion_ratio=0.0)
        except Exception as exc:
            logger.warning("Depth peeling unavailable: %s", exc)

    return plotter


def add_ghost_reference_mesh(
    plotter: pv.Plotter,
    mesh: pv.PolyData | pv.DataSet,
    color: str = "#b8c0cc",
    opacity: float = 0.28,
) -> None:
    """Add undeformed reference geometry as an elegant, neutral ghost overlay."""
    # Ensure smooth surface normals
    if isinstance(mesh, pv.PolyData) and "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals(point_normals=True, feature_angle=60.0)

    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.65,
        specular=0.05,
        show_scalar_bar=False,
    )


def add_deformed_scalar_mesh(
    plotter: pv.Plotter,
    mesh: pv.PolyData | pv.DataSet,
    scalars: str | np.ndarray,
    cmap: str = "viridis",
    clim: tuple[float, float] = (0.0, 1.0),
    ambient: float = 0.28,
    diffuse: float = 0.82,
    specular: float = 0.12,
) -> None:
    """Add deformed mesh colored by continuous scalar field with high fidelity."""
    if isinstance(mesh, pv.PolyData) and "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals(point_normals=True, feature_angle=60.0)

    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        show_scalar_bar=False,
    )


def configure_publication_camera(
    plotter: pv.Plotter,
    center: Sequence[float],
    view: str = "AP",
    distance: float = 700.0,
    parallel_scale: float = 195.0,
    custom_cam: tuple[Any, Any, Any] | None = None,
) -> None:
    """Set standardized camera orientation and parallel projection.

    Views supported:
    - 'AP': Anterior -> Posterior (looking +Y, up +Z)
    - 'PA': Posterior -> Anterior (looking -Y, up +Z)
    - 'CC': Cranial -> Caudal / Superior (looking -Z, up +Y)
    - 'CA': Caudal -> Cranial / Inferior (looking +Z, up -Y)
    - 'OBLIQUE': Anterosuperior 3/4 perspective
    """
    center_arr = np.asarray(center, dtype=float)

    if custom_cam is not None:
        plotter.camera_position = custom_cam
    elif view == "AP":
        cam_pos = center_arr + distance * np.array([0.0, -1.0, 0.0])
        plotter.camera_position = [cam_pos, center_arr, [0.0, 0.0, 1.0]]
    elif view == "PA":
        cam_pos = center_arr + distance * np.array([0.0, 1.0, 0.0])
        plotter.camera_position = [cam_pos, center_arr, [0.0, 0.0, 1.0]]
    elif view in ("CC", "CRANIAL", "SUPERIOR"):
        cam_pos = center_arr + distance * np.array([0.0, 0.0, 1.0])
        plotter.camera_position = [cam_pos, center_arr, [0.0, 1.0, 0.0]]
    elif view in ("CA", "CAUDAL", "INFERIOR"):
        cam_pos = center_arr + distance * np.array([0.0, 0.0, -1.0])
        plotter.camera_position = [cam_pos, center_arr, [0.0, -1.0, 0.0]]
    elif view == "OBLIQUE":
        cam_pos = [center_arr[0] + 15.0, center_arr[1] - 390.0, center_arr[2] + 240.0]
        plotter.camera_position = [cam_pos, center_arr, (0.0, 0.4, 0.9)]
    else:
        raise ValueError(f"Unknown standard view: {view}")

    plotter.enable_parallel_projection()
    if parallel_scale is not None:
        plotter.camera.parallel_scale = parallel_scale


def crop_image_whitespace(
    img: np.ndarray,
    threshold: int = 245,
    pad: int = 8,
) -> np.ndarray:
    """Tight-crop whitespace from rendered image with deterministic padding."""
    yy, xx = np.where(np.any(img[:, :, :3] < threshold, axis=2))
    if len(yy) == 0 or len(xx) == 0:
        return img
    ymin, ymax = max(0, yy.min() - pad), min(img.shape[0], yy.max() + pad + 1)
    xmin, xmax = max(0, xx.min() - pad), min(img.shape[1], xx.max() + pad + 1)
    return img[ymin:ymax, xmin:xmax]

