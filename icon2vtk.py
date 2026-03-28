#!/usr/bin/env python3
"""Convert ICON cell-based netCDF data into legacy VTK for ParaView.

The script combines two inputs:

- an ICON data file containing one or more variables on the ``ncells`` grid
- an ICON grid file containing the triangular mesh geometry

The export target is a legacy VTK unstructured grid made of triangles.  ParaView
can read that format directly, which makes it a convenient bridge from ICON's
netCDF representation into an interactive visualization workflow.

Although the core task is simple, there are a few ICON-specific details that
are easy to miss when reading the code:

- ICON stores cell connectivity with 1-based indices, while NumPy uses 0-based
  indexing, so the mesh reader normalizes the connectivity immediately.
- Variables may have extra dimensions such as ``time`` or vertical levels; the
  field reader resolves those to one horizontal slice before export.
- Optional mesh coarsening reconstructs parent triangles from four sibling
  child triangles using ICON parent-child metadata.
- ``plate-carree`` output needs special longitude handling near the dateline so
  individual cells and polylines stay visually continuous.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

DEFAULT_EARTH_RADIUS_M = 6371229.0


class BlankLineHelpFormatter(argparse.HelpFormatter):
    """Argparse formatter that inserts a blank line after each help entry."""

    def _format_action(self, action: argparse.Action) -> str:
        text = super()._format_action(action)
        if text.endswith("\n"):
            return text + "\n"
        return text + "\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=BlankLineHelpFormatter,
        description=(
            "Read ICON netCDF data and grid information to write VTK for "
            "ParaView, or generate coastline/graticule overlays on their own."
        ),
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        help="netCDF file holding the field to export or to inspect with --list-variables",
    )
    parser.add_argument(
        "grid_file",
        nargs="?",
        help="ICON grid file, e.g. icon_grid_*.nc (required for field export)",
    )
    parser.add_argument(
        "variable",
        nargs="?",
        help="Variable name to export (required for field export)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output VTK file path (default: ./<variable>.vtk in the current working directory)",
    )
    parser.add_argument(
        "--time-index",
        default="0",
        help=(
            "Time index or comma-separated list of time indices to export "
            "if a time dimension is present (default: 0)"
        ),
    )
    parser.add_argument(
        "--level-index",
        default="0",
        help=(
            "Level index or comma-separated list of indices for one extra "
            "non-singleton non-cell dimension besides time, such as a "
            "vertical level; singleton extra dimensions are selected "
            "automatically (default: 0)"
        ),
    )
    parser.add_argument(
        "--coarsen-level",
        type=int,
        default=0,
        help=(
            "Coarsen the field mesh by N ICON refinement levels using "
            "parent-child metadata and sibling averaging (default: 0)"
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help=(
            "Sphere radius in meters used for spherical output and as the "
            "coordinate scale for plate-carree output. Defaults to "
            f"{DEFAULT_EARTH_RADIUS_M:.16g}."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("sphere", "plate-carree"),
        default="sphere",
        help=(
            "Geometry projection for the exported mesh and overlays "
            "(default: sphere)"
        ),
    )
    parser.add_argument(
        "--plate-carree-seam-mode",
        choices=("wrap", "clip"),
        default="wrap",
        help=(
            "How plate-carree cells near the dateline are handled: "
            "'wrap' keeps triangles contiguous with slight overhang, "
            "'clip' clamps longitudes into [-180, 180] for display "
            "(default: wrap)"
        ),
    )
    parser.add_argument(
        "--field-radius-offset",
        type=float,
        default=0.0,
        help=(
            "Add this offset to the exported field geometry. For sphere this "
            "increases the sphere radius; for flat projections it becomes a "
            "constant z offset."
        ),
    )
    parser.add_argument(
        "--coastline-output",
        help=(
            "Optional VTK output path for coastline polylines derived from "
            "Cartopy/Natural Earth."
        ),
    )
    parser.add_argument(
        "--coastline-resolution",
        choices=("110m", "50m", "10m"),
        default="110m",
        help="Natural Earth coastline resolution to export (default: 110m)",
    )
    parser.add_argument(
        "--coastline-radius-offset",
        type=float,
        default=10000.0,
        help=(
            "Add this offset to the exported coastline geometry. For sphere "
            "this increases the sphere radius; for flat projections it "
            "becomes a constant z offset (default: 10000)."
        ),
    )
    parser.add_argument(
        "--river-output",
        help=(
            "Optional VTK output path for river polylines derived from "
            "Cartopy/Natural Earth."
        ),
    )
    parser.add_argument(
        "--river-resolution",
        choices=("110m", "50m", "10m"),
        default="110m",
        help="Natural Earth river resolution to export (default: 110m)",
    )
    parser.add_argument(
        "--river-radius-offset",
        type=float,
        default=13000.0,
        help=(
            "Add this offset to the exported river geometry. For sphere "
            "this increases the sphere radius; for flat projections it "
            "becomes a constant z offset (default: 13000)."
        ),
    )
    parser.add_argument(
        "--country-output",
        help=(
            "Optional VTK output path for country boundary polylines derived "
            "from Cartopy/Natural Earth."
        ),
    )
    parser.add_argument(
        "--country-resolution",
        choices=("110m", "50m", "10m"),
        default="110m",
        help="Natural Earth country boundary resolution to export (default: 110m)",
    )
    parser.add_argument(
        "--country-radius-offset",
        type=float,
        default=16000.0,
        help=(
            "Add this offset to the exported country boundary geometry. For "
            "sphere this increases the sphere radius; for flat projections it "
            "becomes a constant z offset (default: 16000)."
        ),
    )
    parser.add_argument(
        "--province-output",
        help=(
            "Optional VTK output path for state or province boundary "
            "polylines derived from Cartopy/Natural Earth."
        ),
    )
    parser.add_argument(
        "--province-resolution",
        choices=("110m", "50m", "10m"),
        default="110m",
        help="Natural Earth province boundary resolution to export (default: 110m)",
    )
    parser.add_argument(
        "--province-radius-offset",
        type=float,
        default=16000.0,
        help=(
            "Add this offset to the exported province boundary geometry. For "
            "sphere this increases the sphere radius; for flat projections it "
            "becomes a constant z offset (default: 16000)."
        ),
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help=(
            "Restrict output to a lon/lat box in degrees. Cells are selected "
            "by ICON cell center; coastlines and graticules are clipped to "
            "the same box."
        ),
    )
    parser.add_argument(
        "--circle",
        nargs=3,
        type=float,
        metavar=("LON_CENTER", "LAT_CENTER", "RADIUS_KM"),
        help=(
            "Restrict output to a circle given by center lon/lat in degrees "
            "and radius in km. Cells are selected by ICON cell center; "
            "coastlines and graticules are filtered to the same region."
        ),
    )
    parser.add_argument(
        "--graticule-output",
        help=(
            "Optional VTK output path for longitude/latitude graticule "
            "lines in the selected projection."
        ),
    )
    parser.add_argument(
        "--graticule-spacing",
        nargs=2,
        type=float,
        metavar=("DLON", "DLAT"),
        default=(30.0, 15.0),
        help="Longitude/latitude spacing in degrees for graticule lines (default: 30 15)",
    )
    parser.add_argument(
        "--graticule-radius-offset",
        type=float,
        default=19000.0,
        help=(
            "Add this offset to the exported graticule geometry. For sphere "
            "this increases the sphere radius; for flat projections it "
            "becomes a constant z offset (default: 19000)."
        ),
    )
    parser.add_argument(
        "--vtk-format",
        choices=("ascii", "binary"),
        default="binary",
        help="Legacy VTK output format (default: binary)",
    )
    parser.add_argument(
        "--vtk-precision",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "Floating-point precision used for VTK point coordinates and field "
            "scalars (default: float32)"
        ),
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List variables in the netCDF data file and exit",
    )
    return parser.parse_args()


def choose_output_path(args: argparse.Namespace) -> Path:
    """Return the explicit output path or fall back to ``<variable>.vtk``."""
    if args.output:
        return Path(args.output)
    return Path(f"{args.variable}.vtk")


def parse_index_list(spec: str, option_name: str) -> list[int]:
    """Parse a comma-separated CLI index list into a non-empty integer list."""
    values: list[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"{option_name} contains an empty entry")
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(
                f"{option_name} must be an integer or comma-separated integers, got {spec!r}"
            ) from exc
        if value < 0:
            raise ValueError(f"{option_name} must contain only non-negative indices")
        values.append(value)
    if not values:
        raise ValueError(f"{option_name} must not be empty")
    return values


def build_output_path(
    base_output_path: Path,
    time_index: int | None,
    level_index: int | None,
    is_batch: bool,
) -> Path:
    """Return the output path for one export slice."""
    if not is_batch:
        return base_output_path

    suffix_parts = []
    if time_index is not None:
        suffix_parts.append(f"t{time_index}")
    if level_index is not None:
        suffix_parts.append(f"l{level_index}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    return base_output_path.with_name(
        f"{base_output_path.stem}{suffix}{base_output_path.suffix}"
    )


def bbox_contains(
    lon_deg: np.ndarray, lat_deg: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray:
    """Return a boolean mask for points whose lon/lat fall inside a bbox.

    The bbox may cross the dateline. In that case ``lon_min > lon_max`` and the
    longitude test is interpreted as two intervals joined across +/-180 degrees.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mask = (lat_deg >= lat_min) & (lat_deg <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (lon_deg >= lon_min) & (lon_deg <= lon_max)
    else:
        lon_mask = (lon_deg >= lon_min) | (lon_deg <= lon_max)
    return lon_mask & lat_mask


def normalize_lon(lon_deg: np.ndarray) -> np.ndarray:
    """Map longitudes into the conventional ``[-180, 180)`` range."""
    return ((lon_deg + 180.0) % 360.0) - 180.0


def great_circle_distance_km(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    radius_m: float,
) -> np.ndarray:
    """Compute spherical great-circle distances with the haversine formula."""
    lon1 = np.deg2rad(normalize_lon(lon_deg))
    lat1 = np.deg2rad(lat_deg)
    lon2 = math.radians(((center_lon_deg + 180.0) % 360.0) - 180.0)
    lat2 = math.radians(center_lat_deg)
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(lat1) * math.cos(lat2) * sin_dlon**2
    central_angle = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return (radius_m * central_angle) / 1000.0


def circle_contains(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    circle: tuple[float, float, float],
    sphere_radius_m: float,
) -> np.ndarray:
    """Return a boolean mask for points lying inside the requested circle."""
    center_lon, center_lat, radius_km = circle
    distance_km = great_circle_distance_km(
        lon_deg,
        lat_deg,
        center_lon,
        center_lat,
        sphere_radius_m,
    )
    return distance_km <= radius_km


def circle_to_bbox(
    circle: tuple[float, float, float],
    sphere_radius_m: float,
) -> tuple[float, float, float, float]:
    """Build a coarse lon/lat bbox that fully contains the requested circle.

    This is used as a cheap first spatial filter for overlays.  It is not the
    exact circle boundary; exact filtering is applied separately where needed.
    """
    center_lon, center_lat, radius_km = circle
    angular_radius_deg = math.degrees((radius_km * 1000.0) / sphere_radius_m)
    lat_min = max(-90.0, center_lat - angular_radius_deg)
    lat_max = min(90.0, center_lat + angular_radius_deg)
    cos_lat = math.cos(math.radians(center_lat))
    if abs(cos_lat) < 1e-12 or lat_min <= -90.0 or lat_max >= 90.0:
        return (-180.0, lat_min, 180.0, lat_max)
    lon_delta = min(180.0, angular_radius_deg / abs(cos_lat))
    lon_min = normalize_lon(np.asarray([center_lon - lon_delta]))[0]
    lon_max = normalize_lon(np.asarray([center_lon + lon_delta]))[0]
    return (float(lon_min), lat_min, float(lon_max), lat_max)


def subset_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    cell_mask: np.ndarray,
    region_description: str = "requested region",
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only selected cells and compact the vertex array accordingly."""
    selected_cells = cells[cell_mask]
    if selected_cells.size == 0:
        raise ValueError(f"No ICON cells fall inside the {region_description}")

    # VTK should only see vertices that are still referenced by the subset.  The
    # ``inverse`` mapping rewrites the old vertex ids into the compacted array.
    used_vertices, inverse = np.unique(selected_cells.ravel(), return_inverse=True)
    subset_points = points[used_vertices]
    subset_cells = inverse.reshape(selected_cells.shape)
    return subset_points, subset_cells


def cell_center_lonlat(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lon/lat cell centers from the current triangle geometry."""
    centroids = np.mean(points[cells], axis=1)
    return xyz_to_lonlat(centroids)


def subset_field_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    radius: float,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset an already-built field mesh by coarse cell-center lon/lat."""
    if bbox is not None and circle is not None:
        raise ValueError("Use either --bbox or --circle, not both")
    if bbox is None and circle is None:
        return points, cells, values

    lon_deg, lat_deg = cell_center_lonlat(points, cells)
    if bbox is not None:
        cell_mask = bbox_contains(lon_deg, lat_deg, bbox)
        region_description = "requested bounding box"
    else:
        cell_mask = circle_contains(lon_deg, lat_deg, circle, radius)
        region_description = "requested circle"

    subset_points, subset_cells = subset_mesh(
        points, cells, cell_mask, region_description
    )
    return subset_points, subset_cells, values[cell_mask]


def compact_cells(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove unused vertices after topology-changing operations such as coarsening."""
    if cells.size == 0:
        raise ValueError("No ICON cells remain after coarsening")
    used_vertices, inverse = np.unique(cells.ravel(), return_inverse=True)
    return points[used_vertices], inverse.reshape(cells.shape)


def average_values_rows(values: np.ndarray) -> np.ndarray:
    """Average each row over finite values and propagate all-NaN rows as ``nan``."""
    finite_mask = np.isfinite(values)
    finite_counts = finite_mask.sum(axis=1)
    finite_sums = np.where(finite_mask, values, 0.0).sum(axis=1, dtype=np.float64)
    means = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(
        finite_sums,
        finite_counts,
        out=means,
        where=finite_counts > 0,
    )
    return means


def order_triangle_vertices_batch(
    points: np.ndarray, vertex_ids: np.ndarray
) -> np.ndarray:
    """Vectorized variant of ``order_triangle_vertices`` for many triangles."""
    triangle_points = points[vertex_ids]
    centroid = triangle_points.mean(axis=1)
    ordered = vertex_ids.copy()

    centroid_norm = np.linalg.norm(centroid, axis=1)
    valid = centroid_norm > 0.0
    if not np.any(valid):
        return ordered

    normal = np.zeros_like(centroid)
    normal[valid] = centroid[valid] / centroid_norm[valid, None]

    first_vec = triangle_points[:, 0, :] - centroid
    first_norm = np.linalg.norm(first_vec, axis=1)
    valid &= first_norm > 0.0
    if not np.any(valid):
        return ordered

    axis_x = np.zeros_like(first_vec)
    axis_x[valid] = first_vec[valid] / first_norm[valid, None]

    axis_y = np.cross(normal, axis_x)
    axis_y_norm = np.linalg.norm(axis_y, axis=1)
    valid &= axis_y_norm > 0.0
    if not np.any(valid):
        return ordered

    axis_y[valid] = axis_y[valid] / axis_y_norm[valid, None]

    rel = triangle_points - centroid[:, None, :]
    angles = np.empty((vertex_ids.shape[0], 3), dtype=np.float64)
    angles.fill(0.0)
    angles[valid] = np.arctan2(
        np.einsum("nij,nj->ni", rel[valid], axis_y[valid]),
        np.einsum("nij,nj->ni", rel[valid], axis_x[valid]),
    )

    order = np.argsort(angles, axis=1)
    ordered[valid] = np.take_along_axis(vertex_ids[valid], order[valid], axis=1)

    ordered_points = points[ordered[valid]]
    orientation = np.einsum(
        "ij,ij->i",
        np.cross(
            ordered_points[:, 1, :] - ordered_points[:, 0, :],
            ordered_points[:, 2, :] - ordered_points[:, 0, :],
        ),
        centroid[valid],
    )
    flip_mask = orientation < 0.0
    if np.any(flip_mask):
        flipped = ordered[valid][flip_mask].copy()
        flipped[:, [1, 2]] = flipped[:, [2, 1]]
        ordered_valid = ordered[valid].copy()
        ordered_valid[flip_mask] = flipped
        ordered[valid] = ordered_valid
    return ordered


def coarsen_one_level(
    points: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    cell_ids: np.ndarray,
    cell_levels: np.ndarray,
    candidate_level: int,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Collapse one ICON refinement level where complete sibling families exist.

    ICON refinement splits one parent triangle into four child triangles.  If
    all four siblings are present, we reconstruct the parent triangle from the
    three outer vertices and assign it the mean of the four child values.

    Incomplete families are intentionally left untouched.  This matters for
    subsets near the selection boundary, where only some children may survive.
    """
    if parent_ids.shape[0] != cells.shape[0]:
        raise ValueError("parent id array length does not match the selected cells")
    if cell_ids.shape[0] != cells.shape[0] or cell_levels.shape[0] != cells.shape[0]:
        raise ValueError("cell metadata length does not match the selected cells")

    candidate_mask = cell_levels == candidate_level
    candidate_indices = np.flatnonzero(candidate_mask & (parent_ids > 0))
    if candidate_indices.size == 0:
        return points, cells, values, cell_ids, cell_levels, False

    candidate_parent_ids = parent_ids[candidate_indices]
    sort_order = np.argsort(candidate_parent_ids, kind="mergesort")
    sorted_parent_ids = candidate_parent_ids[sort_order]
    sorted_indices = candidate_indices[sort_order]
    unique_parent_ids, family_starts, family_counts = np.unique(
        sorted_parent_ids, return_index=True, return_counts=True
    )
    complete_family_mask = family_counts == 4
    if not np.any(complete_family_mask):
        return points, cells, values, cell_ids, cell_levels, False

    family_starts = family_starts[complete_family_mask]
    complete_parent_ids = unique_parent_ids[complete_family_mask].astype(np.int64)
    sibling_indices = sorted_indices[family_starts[:, None] + np.arange(4)]
    sibling_cells = cells[sibling_indices]
    sorted_vertices = np.sort(sibling_cells.reshape(sibling_indices.shape[0], 12), axis=1)
    prev_vertices = np.pad(sorted_vertices[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    next_vertices = np.pad(sorted_vertices[:, 1:], ((0, 0), (0, 1)), constant_values=-1)
    parent_vertex_mask = (sorted_vertices != prev_vertices) & (
        sorted_vertices != next_vertices
    )
    valid_family_mask = parent_vertex_mask.sum(axis=1) == 3
    if not np.any(valid_family_mask):
        return points, cells, values, cell_ids, cell_levels, False

    sibling_indices = sibling_indices[valid_family_mask]
    sibling_cells = sibling_cells[valid_family_mask]
    complete_parent_ids = complete_parent_ids[valid_family_mask]
    parent_vertices = sorted_vertices[valid_family_mask][parent_vertex_mask[valid_family_mask]].reshape(-1, 3)

    coarsened_cells = order_triangle_vertices_batch(points, parent_vertices.astype(np.int64))
    coarsened_values = average_values_rows(values[sibling_indices])
    coarsened_ids = complete_parent_ids
    coarsened_levels = np.full(
        complete_parent_ids.shape[0], candidate_level + 1, dtype=np.int64
    )
    keep_child_mask = np.ones(cells.shape[0], dtype=bool)
    keep_child_mask[sibling_indices.reshape(-1)] = False

    # Keep all cells that were not replaced by a reconstructed parent.
    remaining_cells = cells[keep_child_mask]
    remaining_values = values[keep_child_mask]
    remaining_ids = cell_ids[keep_child_mask]
    remaining_levels = cell_levels[keep_child_mask]

    if coarsened_cells.shape[0] > 0:
        combined_cells = np.vstack((remaining_cells, coarsened_cells))
        combined_values = np.concatenate(
            (remaining_values, coarsened_values)
        )
        combined_ids = np.concatenate((remaining_ids, coarsened_ids))
        combined_levels = np.concatenate(
            (remaining_levels, coarsened_levels)
        )
    else:
        combined_cells = remaining_cells
        combined_values = remaining_values
        combined_ids = remaining_ids
        combined_levels = remaining_levels

    compact_points, compacted_cells = compact_cells(points, combined_cells)
    return (
        compact_points,
        compacted_cells,
        combined_values,
        combined_ids,
        combined_levels,
        True,
    )


def coarsen_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    parent_cell_index: np.ndarray | None,
    coarsen_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply ``coarsen_one_level`` repeatedly up to the requested depth."""
    if coarsen_level < 0:
        raise ValueError("Coarsen level must be non-negative")
    if coarsen_level == 0:
        return points, cells, values, 0
    if parent_cell_index is None:
        raise ValueError(
            "Grid file does not provide parent_cell_index needed for --coarsen-level"
        )
    if parent_cell_index.shape[0] != cells.shape[0]:
        raise ValueError("parent_cell_index length does not match the selected cells")

    current_points = points
    current_cells = cells
    current_values = values
    # ICON metadata uses 1-based cell ids, so we mirror that convention here.
    current_cell_ids = np.arange(1, cells.shape[0] + 1, dtype=np.int64)
    current_cell_levels = np.zeros(cells.shape[0], dtype=np.int64)
    current_parent_ids = np.asarray(parent_cell_index, dtype=np.int64)
    applied_levels = 0

    for level in range(coarsen_level):
        (
            current_points,
            current_cells,
            current_values,
            current_cell_ids,
            current_cell_levels,
            changed,
        ) = coarsen_one_level(
            current_points,
            current_cells,
            current_values,
            current_cell_ids,
            current_cell_levels,
            level,
            current_parent_ids,
        )
        if not changed:
            break
        applied_levels += 1
        # After the first collapse we no longer have original grid metadata for
        # the synthetic coarse mesh.  ICON's regular 4:1 refinement pattern lets
        # us infer the next parent ids from the current 1-based cell ids.
        current_parent_ids = ((current_cell_ids - 1) // 4) + 1

    return current_points, current_cells, current_values, applied_levels


def read_mesh(
    grid_path: Path,
    radius_override: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read the full ICON triangular mesh and optional coarsening metadata."""
    with Dataset(grid_path) as ds:
        required = [
            "vertex_of_cell",
            "cartesian_x_vertices",
            "cartesian_y_vertices",
            "cartesian_z_vertices",
        ]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(
                f"Grid file is missing required variables: {', '.join(missing)}"
            )

        radius = radius_override
        if radius is None:
            radius = DEFAULT_EARTH_RADIUS_M

        x = (
            np.asarray(ds.variables["cartesian_x_vertices"][:], dtype=np.float64)
            * radius
        )
        y = (
            np.asarray(ds.variables["cartesian_y_vertices"][:], dtype=np.float64)
            * radius
        )
        z = (
            np.asarray(ds.variables["cartesian_z_vertices"][:], dtype=np.float64)
            * radius
        )
        points = np.column_stack((x, y, z))

        connectivity = np.asarray(ds.variables["vertex_of_cell"][:], dtype=np.int64)
        if connectivity.ndim != 2 or connectivity.shape[0] != 3:
            raise ValueError(
                "Expected vertex_of_cell to have shape (3, ncells) for triangular ICON cells"
            )

        # ICON stores vertex ids as 1-based indices with shape (3, ncells).
        # VTK/NumPy expect per-cell rows with 0-based indices.
        cells = connectivity.T - 1
        if np.any(cells < 0):
            raise ValueError("vertex_of_cell appears not to be 1-based as expected")
        parent_cell_index = None
        if "parent_cell_index" in ds.variables:
            parent_cell_index = np.asarray(
                ds.variables["parent_cell_index"][:], dtype=np.int64
            )
            if parent_cell_index.shape != (cells.shape[0],):
                raise ValueError(
                    "Expected parent_cell_index to have one entry per ICON cell"
                )
    return points, cells, parent_cell_index


def resolve_radius(radius_override: float | None) -> float:
    """Resolve the sphere radius, using a fixed default unless overridden."""
    if radius_override is not None:
        return float(radius_override)
    return DEFAULT_EARTH_RADIUS_M


def ensure_variable_exists(data_path: Path, variable_name: str) -> None:
    """Fail early if the requested variable is not present in the data file."""
    with Dataset(data_path) as ds:
        if variable_name in ds.variables:
            return
        available = ", ".join(ds.variables.keys())
        raise ValueError(
            f"Variable {variable_name!r} not found. Available variables: {available}"
        )


def read_field(
    data_path: Path,
    variable_name: str,
    time_index: int,
    level_index: int,
    expected_ncells: int,
) -> tuple[np.ndarray, str | None, str, bool, bool]:
    """Read one ICON variable and resolve it to a 1-D array over ``ncells``.

    The variable may depend on ``time`` and on one additional non-cell
    dimension such as ``height``.  Every non-``ncells`` dimension is reduced to
    a single index so the result always matches the horizontal mesh.
    """
    with Dataset(data_path) as ds:
        if variable_name not in ds.variables:
            available = ", ".join(ds.variables.keys())
            raise ValueError(
                f"Variable {variable_name!r} not found. Available variables: {available}"
            )

        var = ds.variables[variable_name]
        dims = var.dimensions
        shape = var.shape
        if "ncells" not in dims:
            raise ValueError(
                f"Variable {variable_name!r} is not cell-based; dimensions are {dims}"
            )

        selection: list[int | slice] = []
        used_level_index = False
        used_time_index = False
        non_singleton_extra_dims = [
            dim_name
            for dim_name, dim_size in zip(dims, shape)
            if dim_name not in {"time", "ncells"} and dim_size > 1
        ]
        if len(non_singleton_extra_dims) > 1:
            raise ValueError(
                f"Variable {variable_name!r} has multiple non-singleton non-cell dimensions "
                f"{tuple(non_singleton_extra_dims)}; only one such dimension is supported"
            )
        for dim_name, dim_size in zip(dims, shape):
            if dim_name == "ncells":
                selection.append(slice(None))
            elif dim_name == "time":
                if not 0 <= time_index < dim_size:
                    raise IndexError(
                        f"time index {time_index} out of range for size {dim_size}"
                    )
                selection.append(time_index)
                used_time_index = True
            else:
                # ICON files often contain singleton helper dimensions such as
                # ``height_2m`` or ``height_10m``. Those are selected
                # automatically, while real multi-level dimensions use the
                # requested ``level_index``.
                idx = 0 if dim_size == 1 else level_index
                if not 0 <= idx < dim_size:
                    raise IndexError(
                        f"level index {level_index} out of range for dimension "
                        f"{dim_name!r} of size {dim_size}"
                    )
                selection.append(idx)
                if dim_size > 1:
                    used_level_index = True

        data = np.ma.asarray(var[tuple(selection)])
        values = np.ma.filled(data, np.nan).astype(np.float64, copy=False)
        values = np.ravel(values)

        if values.size != expected_ncells:
            raise ValueError(
                f"Selected data has {values.size} values, expected {expected_ncells} cells"
            )

        units = getattr(var, "units", None)
        long_name = getattr(var, "long_name", variable_name)
        slice_parts = []
        if used_time_index:
            slice_parts.append(f"time-index={time_index}")
        if used_level_index:
            slice_parts.append(f"level-index={level_index}")
        if slice_parts:
            long_name = f"{long_name} ({', '.join(slice_parts)})"
        return values, units, long_name, used_time_index, used_level_index


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    """Compute a small numeric summary for the exported slice."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "nan_count": int(values.size),
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
        }

    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nan_count": int(values.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def format_shape(shape: tuple[int, ...]) -> str:
    """Format a netCDF variable shape for human-readable output."""
    return "(" + ", ".join(str(v) for v in shape) + ")"


def format_scalar_value(value: object) -> str:
    """Format one coordinate value for compact user-facing output."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        iso_value = value.isoformat()
        return iso_value.replace("+00:00", "Z")
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.16g}"
    return str(value)


def format_indexed_value_sequence(
    dim_name: str,
    values: list[object],
    units: str | None,
    display_units: bool,
    max_items: int = 12,
) -> list[str]:
    """Format a coordinate vector as one ``name[index] = value`` entry per line."""
    unit_suffix = f" {units}" if display_units and units else ""
    indexed_values = [
        f"{dim_name}[{index}] = {format_scalar_value(value)}{unit_suffix}"
        for index, value in enumerate(values)
    ]
    if len(indexed_values) <= max_items:
        return indexed_values
    head_count = max_items // 2
    tail_count = max_items - head_count
    return indexed_values[:head_count] + ["..."] + indexed_values[-tail_count:]


def read_dimension_values(
    ds: Dataset, dim_name: str
) -> tuple[list[object], str | None] | None:
    """Return formatted coordinate values for a 1-D dimension variable when available."""
    if dim_name not in ds.variables:
        return None
    coord_var = ds.variables[dim_name]
    if coord_var.dimensions != (dim_name,):
        return None

    values = np.asarray(coord_var[:]).reshape(-1)
    units = getattr(coord_var, "units", None)
    standard_name = getattr(coord_var, "standard_name", None)

    if (standard_name == "time" or dim_name == "time") and units:
        calendar = getattr(coord_var, "calendar", "standard")
        converted = num2date(values, units=units, calendar=calendar)
        return list(np.asarray(converted, dtype=object).reshape(-1)), units

    return [
        value.item() if isinstance(value, np.generic) else value for value in values
    ], units


def is_summary_coordinate(coord_var, dim_name: str) -> bool:
    """Return whether a dimension should appear in the coordinate summary."""
    standard_name = getattr(coord_var, "standard_name", None)
    is_time = standard_name == "time" or dim_name == "time"
    is_vertical = (
        standard_name in {"height", "altitude", "air_pressure", "model_level_number"}
        or dim_name.startswith("height")
        or dim_name in {"height", "lev", "level", "levels", "plev", "depth"}
    )
    return is_time or is_vertical


def collect_coordinate_summaries(ds: Dataset) -> list[list[str]]:
    """Collect shared time and vertical coordinate values present in the file."""
    coordinate_parts: list[list[str]] = []
    for dim_name in ds.dimensions:
        if dim_name not in ds.variables:
            continue
        coord_var = ds.variables[dim_name]
        if not is_summary_coordinate(coord_var, dim_name):
            continue
        dimension_values = read_dimension_values(ds, dim_name)
        if dimension_values is None:
            continue
        values, units = dimension_values
        coord_var = ds.variables[dim_name]
        display_units = not (
            getattr(coord_var, "standard_name", None) == "time" or dim_name == "time"
        )
        coordinate_parts.append(
            format_indexed_value_sequence(dim_name, values, units, display_units)
        )
    return coordinate_parts


def format_duration(seconds: float) -> str:
    """Format a short wall-clock duration for user-facing logs."""
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    return f"{seconds:.3f} s"


def log_message(message: str) -> None:
    """Print a concise progress message."""
    print(message, flush=True)


def quote_metadata_text(value: str) -> str:
    """Return a double-quoted metadata string for display."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def list_variables(data_path: Path) -> None:
    """Print a compact overview of variables contained in the data file."""
    with Dataset(data_path) as ds:
        coordinate_summaries = collect_coordinate_summaries(ds)
        print()
        print(f"Variables in {data_path}:")
        print()
        for name, var in ds.variables.items():
            dims = ", ".join(var.dimensions) if var.dimensions else "-"
            shape = format_shape(var.shape)
            units = getattr(var, "units", "-")
            long_name = getattr(var, "long_name", "-")
            standard_name = getattr(var, "standard_name", "-")
            dtype = str(var.dtype)
            grid_type = getattr(var, "CDI_grid_type", getattr(var, "grid_type", "-"))
            metadata_parts = []
            if long_name and long_name != "-":
                metadata_parts.append(f"long_name={quote_metadata_text(long_name)}")
            std_text = (
                f"standard_name={quote_metadata_text(standard_name)}"
                if standard_name and standard_name != "-"
                else None
            )
            if std_text is not None:
                metadata_parts.append(std_text)
            unit_text = f"; units=[{units}]" if units and units != "-" else ""
            metadata_text = "; ".join(metadata_parts) if metadata_parts else "-"
            print(f"- {name}: {metadata_text}{unit_text}")
            print(f"    dims={dims}; shape={shape}; grid={grid_type}; dtype={dtype}")
            print()
        if coordinate_summaries:
            print("Coordinate values:")
            print()
            for summary_lines in coordinate_summaries:
                for line in summary_lines:
                    print(f"    {line}")
                print()


def write_header(fh, title: str, dataset_type: str, vtk_format: str) -> None:
    """Write the common legacy VTK file header."""
    fh.write(b"# vtk DataFile Version 3.0\n")
    fh.write(title[:255].encode("ascii", errors="replace") + b"\n")
    fh.write(vtk_format.upper().encode("ascii") + b"\n")
    fh.write(f"DATASET {dataset_type}\n".encode("ascii"))


def write_numeric_array(
    fh,
    array: np.ndarray,
    vtk_format: str,
    trailing_newline: bool = True,
) -> None:
    """Write a numeric NumPy array in legacy VTK ASCII or binary form.

    Legacy binary VTK expects big-endian byte order.  NumPy arrays are normally
    native-endian, so the binary branch converts explicitly before writing.
    """
    if vtk_format == "ascii":
        if array.ndim == 1:
            for value in array:
                if np.issubdtype(array.dtype, np.integer):
                    fh.write(f"{int(value)}\n".encode("ascii"))
                else:
                    if math.isnan(float(value)):
                        fh.write(b"nan\n")
                    else:
                        fh.write(f"{float(value):.16g}\n".encode("ascii"))
            return
        for row in array:
            if np.issubdtype(array.dtype, np.integer):
                line = " ".join(str(int(v)) for v in row)
            else:
                parts = []
                for v in row:
                    fv = float(v)
                    parts.append("nan" if math.isnan(fv) else f"{fv:.16g}")
                line = " ".join(parts)
            fh.write(line.encode("ascii") + b"\n")
        return

    be_array = np.asarray(array, dtype=array.dtype.newbyteorder(">"))
    fh.write(be_array.tobytes(order="C"))
    if trailing_newline:
        fh.write(b"\n")


def vtk_precision_spec(vtk_precision: str) -> tuple[np.dtype, str]:
    """Map a CLI precision selection to NumPy and VTK type names."""
    if vtk_precision == "float32":
        return np.dtype(np.float32), "float"
    if vtk_precision == "float64":
        return np.dtype(np.float64), "double"
    raise ValueError(f"Unsupported VTK precision {vtk_precision!r}")


def write_triangle_cells(
    fh,
    cells: np.ndarray,
    vtk_format: str,
    chunk_size: int = 1_000_000,
) -> None:
    """Write legacy VTK triangle connectivity without materializing one giant array."""
    cells_int = np.asarray(cells, dtype=np.int32)
    if vtk_format == "ascii":
        for start in range(0, cells_int.shape[0], chunk_size):
            stop = min(start + chunk_size, cells_int.shape[0])
            for row in cells_int[start:stop]:
                fh.write(
                    f"3 {int(row[0])} {int(row[1])} {int(row[2])}\n".encode("ascii")
                )
        return

    chunk = np.empty((min(chunk_size, cells_int.shape[0]), 4), dtype=np.int32)
    chunk[:, 0] = 3
    for start in range(0, cells_int.shape[0], chunk_size):
        stop = min(start + chunk_size, cells_int.shape[0])
        size = stop - start
        chunk_view = chunk[:size]
        chunk_view[:, 1:] = cells_int[start:stop]
        write_numeric_array(fh, chunk_view, vtk_format, trailing_newline=False)
    fh.write(b"\n")


def write_constant_cell_types(
    fh,
    ncells: int,
    vtk_format: str,
    cell_type: int = 5,
    chunk_size: int = 4_000_000,
) -> None:
    """Write repeated VTK cell types in chunks to avoid a huge temporary array."""
    if vtk_format == "ascii":
        line = f"{cell_type}\n".encode("ascii")
        for _ in range(ncells):
            fh.write(line)
        return

    chunk = np.full(min(chunk_size, ncells), cell_type, dtype=np.int32)
    for start in range(0, ncells, chunk_size):
        stop = min(start + chunk_size, ncells)
        write_numeric_array(
            fh,
            chunk[: stop - start],
            vtk_format,
            trailing_newline=False,
        )
    fh.write(b"\n")


def write_legacy_vtk(
    output_path: Path,
    points: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    variable_name: str,
    title: str,
    units: str | None,
    vtk_format: str,
    vtk_precision: str,
) -> None:
    """Write the field mesh as a legacy VTK ``UNSTRUCTURED_GRID``."""
    npoints = points.shape[0]
    ncells = cells.shape[0]
    float_dtype, vtk_float_name = vtk_precision_spec(vtk_precision)

    with output_path.open("wb") as fh:
        header = title
        if units:
            header = f"{header} [{units}]"
        write_header(fh, header, "UNSTRUCTURED_GRID", vtk_format)

        fh.write(f"POINTS {npoints} {vtk_float_name}\n".encode("ascii"))
        write_numeric_array(
            fh, np.asarray(points, dtype=float_dtype), vtk_format
        )

        fh.write(f"CELLS {ncells} {ncells * 4}\n".encode("ascii"))
        # Legacy VTK ``CELLS`` encodes each cell as ``npts id0 id1 id2 ...``.
        # For ICON this is always ``3`` followed by three triangle vertex ids.
        write_triangle_cells(fh, cells, vtk_format)

        fh.write(f"CELL_TYPES {ncells}\n".encode("ascii"))
        write_constant_cell_types(fh, ncells, vtk_format)

        fh.write(f"CELL_DATA {ncells}\n".encode("ascii"))
        fh.write(
            f"SCALARS {sanitize_name(variable_name)} {vtk_float_name} 1\n".encode("ascii")
        )
        fh.write(b"LOOKUP_TABLE default\n")
        write_numeric_array(
            fh, np.asarray(values, dtype=float_dtype), vtk_format
        )


def sanitize_name(name: str) -> str:
    """Map arbitrary variable names to VTK-safe scalar array names."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    return cleaned or "field"


def xyz_to_lonlat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates on a sphere back to lon/lat degrees."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    lon_deg = np.rad2deg(np.arctan2(y, x))
    radius = np.linalg.norm(points, axis=1)
    lat_deg = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    zero_radius = radius == 0.0
    if np.any(zero_radius):
        lat_deg = lat_deg.copy()
        lat_deg[zero_radius] = 0.0
    return lon_deg, lat_deg


def lonlat_to_xyz(
    lon_deg: np.ndarray, lat_deg: np.ndarray, radius: float
) -> np.ndarray:
    """Convert lon/lat degrees to Cartesian coordinates on a sphere."""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat)
    x = radius * cos_lat * np.cos(lon)
    y = radius * cos_lat * np.sin(lon)
    z = radius * np.sin(lat)
    return np.column_stack((x, y, z))


def project_lonlat(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    projection: str,
    radius: float,
    offset: float = 0.0,
) -> np.ndarray:
    """Project geographic coordinates either to the sphere or to plate-carree."""
    if projection == "sphere":
        return lonlat_to_xyz(lon_deg, lat_deg, radius + offset)
    if projection == "plate-carree":
        x = radius * np.deg2rad(lon_deg)
        y = radius * np.deg2rad(lat_deg)
        z = np.full_like(x, offset, dtype=np.float64)
        return np.column_stack((x, y, z))
    raise ValueError(f"Unsupported projection {projection!r}")


def project_xyz(
    points: np.ndarray, projection: str, radius: float, offset: float = 0.0
) -> np.ndarray:
    """Project existing mesh vertices while preserving their current topology."""
    if projection == "sphere":
        if offset == 0.0:
            return points
        if radius == 0.0:
            raise ValueError("Cannot apply a sphere offset when radius is zero")
        scale = (radius + offset) / radius
        return points * scale
    lon_deg, lat_deg = xyz_to_lonlat(points)
    return project_lonlat(lon_deg, lat_deg, projection, radius, offset)


def wrap_longitudes_to_primary_range(lon_deg: np.ndarray) -> np.ndarray:
    """Shift a locally continuous longitude set near the usual display range."""
    wrapped_lon = lon_deg.copy()
    center_lon = float(np.mean(wrapped_lon))
    wrapped_lon -= 360.0 * math.floor((center_lon + 180.0) / 360.0)
    if np.mean(wrapped_lon) > 180.0:
        wrapped_lon -= 360.0
    elif np.mean(wrapped_lon) < -180.0:
        wrapped_lon += 360.0
    return wrapped_lon


def clip_longitudes_to_primary_range(lon_deg: np.ndarray) -> np.ndarray:
    """Clamp longitudes to the visible ``[-180, 180]`` plate-carree range."""
    return np.clip(lon_deg, -180.0, 180.0)


def unwrap_longitudes_for_polyline(lon_deg: np.ndarray) -> np.ndarray:
    """Unwrap a polyline so consecutive points stay geographically adjacent."""
    return np.rad2deg(np.unwrap(np.deg2rad(lon_deg)))


def project_polyline(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    projection: str,
    radius: float,
    offset: float = 0.0,
    plate_carree_seam_mode: str = "wrap",
) -> np.ndarray:
    """Project overlay polylines with seam handling for flat map output."""
    if projection != "plate-carree":
        return project_lonlat(lon_deg, lat_deg, projection, radius, offset)

    adjusted_lon = unwrap_longitudes_for_polyline(lon_deg)
    adjusted_lon = wrap_longitudes_to_primary_range(adjusted_lon)
    if plate_carree_seam_mode == "clip":
        adjusted_lon = clip_longitudes_to_primary_range(adjusted_lon)
    elif plate_carree_seam_mode != "wrap":
        raise ValueError(
            f"Unsupported plate-carree seam mode {plate_carree_seam_mode!r}"
        )
    return project_lonlat(adjusted_lon, lat_deg, projection, radius, offset)


def project_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    projection: str,
    radius: float,
    offset: float = 0.0,
    plate_carree_seam_mode: str = "wrap",
) -> tuple[np.ndarray, np.ndarray]:
    """Project the field mesh into the requested output geometry.

    For ``sphere`` the original connectivity can be reused directly.  For
    ``plate-carree`` each triangle is still projected independently so cells
    crossing the dateline can duplicate vertices and remain visually
    contiguous. The implementation is vectorized across all cells to avoid the
    expensive Python loop used in the original version.
    """
    if projection == "sphere":
        return project_xyz(points, projection, radius, offset), cells

    if projection != "plate-carree":
        raise ValueError(f"Unsupported projection {projection!r}")

    lon_deg, lat_deg = xyz_to_lonlat(points)
    cell_lon = lon_deg[cells]
    cell_lat = lat_deg[cells]

    non_polar_mask = np.abs(cell_lat) < 89.999999
    has_non_polar = np.any(non_polar_mask, axis=1)
    reference_index = np.argmax(non_polar_mask, axis=1)
    reference_lon = np.where(
        has_non_polar,
        cell_lon[np.arange(cells.shape[0]), reference_index],
        cell_lon[:, 0],
    )

    adjusted_lon = (
        reference_lon[:, np.newaxis]
        + ((cell_lon - reference_lon[:, np.newaxis] + 180.0) % 360.0)
        - 180.0
    )

    polar_mask = ~non_polar_mask
    polar_rows = np.any(polar_mask, axis=1) & has_non_polar
    if np.any(polar_rows):
        non_polar_count = np.sum(non_polar_mask, axis=1)
        non_polar_mean_lon = np.divide(
            np.sum(adjusted_lon * non_polar_mask, axis=1),
            non_polar_count,
            out=np.zeros(cells.shape[0], dtype=np.float64),
            where=non_polar_count > 0,
        )
        adjusted_lon = np.where(
            polar_mask & polar_rows[:, np.newaxis],
            non_polar_mean_lon[:, np.newaxis],
            adjusted_lon,
        )

    center_lon = np.mean(adjusted_lon, axis=1)
    wrapped_lon = (
        adjusted_lon - 360.0 * np.floor((center_lon + 180.0) / 360.0)[:, np.newaxis]
    )
    wrapped_mean_lon = np.mean(wrapped_lon, axis=1)
    wrapped_lon = np.where(
        (wrapped_mean_lon > 180.0)[:, np.newaxis],
        wrapped_lon - 360.0,
        wrapped_lon,
    )
    wrapped_lon = np.where(
        (wrapped_mean_lon < -180.0)[:, np.newaxis],
        wrapped_lon + 360.0,
        wrapped_lon,
    )

    if plate_carree_seam_mode == "clip":
        wrapped_lon = clip_longitudes_to_primary_range(wrapped_lon)
    elif plate_carree_seam_mode != "wrap":
        raise ValueError(
            f"Unsupported plate-carree seam mode {plate_carree_seam_mode!r}"
        )

    projected_points = project_lonlat(
        wrapped_lon.reshape(-1),
        cell_lat.reshape(-1),
        projection,
        radius,
        offset,
    )
    projected_cells = np.arange(
        cells.shape[0] * cells.shape[1], dtype=np.int64
    ).reshape(cells.shape)
    return projected_points, projected_cells


def iter_linestring_coords(geometry) -> list[np.ndarray]:
    """Extract coordinate arrays from Shapely line-like geometries."""
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        return [np.asarray(geometry.coords, dtype=np.float64)]
    if geom_type == "MultiLineString":
        return [np.asarray(line.coords, dtype=np.float64) for line in geometry.geoms]
    if geom_type == "GeometryCollection":
        coords = []
        for item in geometry.geoms:
            coords.extend(iter_linestring_coords(item))
        return coords
    return []


def split_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Split a boolean mask into contiguous runs of ``True`` indices."""
    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        return []
    splits = np.where(np.diff(true_indices) != 1)[0] + 1
    return [group for group in np.split(true_indices, splits) if group.size > 0]


def write_coastline_vtk(
    output_path: Path,
    projection: str,
    radius: float,
    resolution: str,
    radius_offset: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    vtk_format: str = "ascii",
    vtk_precision: str = "float32",
    plate_carree_seam_mode: str = "wrap",
) -> int:
    """Write Natural Earth coastlines as legacy VTK ``POLYDATA`` lines."""
    return write_natural_earth_lines_vtk(
        output_path=output_path,
        projection=projection,
        radius=radius,
        resolution=resolution,
        category="physical",
        dataset_name="coastline",
        feature_label="coastline",
        radius_offset=radius_offset,
        bbox=bbox,
        circle=circle,
        vtk_format=vtk_format,
        vtk_precision=vtk_precision,
        plate_carree_seam_mode=plate_carree_seam_mode,
    )


def write_natural_earth_lines_vtk(
    output_path: Path,
    projection: str,
    radius: float,
    resolution: str,
    category: str,
    dataset_name: str,
    feature_label: str,
    radius_offset: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    vtk_format: str = "ascii",
    vtk_precision: str = "float32",
    plate_carree_seam_mode: str = "wrap",
) -> int:
    """Write a Natural Earth line dataset as legacy VTK ``POLYDATA``."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import GeometryCollection, box
    except ImportError as exc:
        raise RuntimeError(
            f"Cartopy and Shapely are required for {feature_label} export but are not available"
        ) from exc

    shapefile_path = shpreader.natural_earth(
        resolution=resolution,
        category=category,
        name=dataset_name,
    )

    if bbox is not None and circle is not None:
        raise RuntimeError("Use either --bbox or --circle, not both")

    bbox_geom = None
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        if lon_min <= lon_max:
            bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
        else:
            bbox_geom = GeometryCollection(
                [
                    box(lon_min, lat_min, 180.0, lat_max),
                    box(-180.0, lat_min, lon_max, lat_max),
                ]
            )
    elif circle is not None:
        lon_min, lat_min, lon_max, lat_max = circle_to_bbox(circle, radius)
        if lon_min <= lon_max:
            bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
        else:
            bbox_geom = GeometryCollection(
                [
                    box(lon_min, lat_min, 180.0, lat_max),
                    box(-180.0, lat_min, lon_max, lat_max),
                ]
            )

    all_points: list[np.ndarray] = []
    line_lengths: list[int] = []
    for record in shpreader.Reader(shapefile_path).records():
        geometry = record.geometry
        if bbox_geom is not None:
            geometry = geometry.intersection(bbox_geom)
            if geometry.is_empty:
                continue
        for coords in iter_linestring_coords(geometry):
            if coords.shape[0] < 2:
                continue
            if circle is not None:
                # Circle filtering is done pointwise on the sampled coastline.
                # Segments that fall outside are split into shorter visible runs.
                point_mask = circle_contains(coords[:, 0], coords[:, 1], circle, radius)
                filtered_segments = [
                    coords[group]
                    for group in split_true_runs(point_mask)
                    if group.size >= 2
                ]
                if not filtered_segments:
                    continue
            else:
                filtered_segments = [coords]
            for segment in filtered_segments:
                projected = project_polyline(
                    segment[:, 0],
                    segment[:, 1],
                    projection,
                    radius,
                    radius_offset,
                    plate_carree_seam_mode,
                )
                all_points.append(projected)
                line_lengths.append(int(projected.shape[0]))

    if not all_points:
        raise RuntimeError(
            f"No {feature_label} geometries were found in the Cartopy dataset"
        )

    points = np.vstack(all_points)
    float_dtype, vtk_float_name = vtk_precision_spec(vtk_precision)
    with output_path.open("wb") as fh:
        write_header(
            fh,
            f"Natural Earth {feature_label}s [{resolution}]",
            "POLYDATA",
            vtk_format,
        )
        fh.write(f"POINTS {points.shape[0]} {vtk_float_name}\n".encode("ascii"))
        write_numeric_array(
            fh, np.asarray(points, dtype=float_dtype), vtk_format
        )
        total_size = sum(length + 1 for length in line_lengths)
        fh.write(f"LINES {len(line_lengths)} {total_size}\n".encode("ascii"))
        line_rows = []
        offset = 0
        for length in line_lengths:
            row = np.empty(length + 1, dtype=np.int32)
            row[0] = length
            row[1:] = np.arange(offset, offset + length, dtype=np.int32)
            line_rows.append(row)
            offset += length
        if vtk_format == "ascii":
            for row in line_rows:
                fh.write((" ".join(str(int(v)) for v in row) + "\n").encode("ascii"))
        else:
            line_blob = np.concatenate(line_rows)
            write_numeric_array(fh, line_blob, vtk_format)

    return len(line_lengths)


def write_river_vtk(
    output_path: Path,
    projection: str,
    radius: float,
    resolution: str,
    radius_offset: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    vtk_format: str = "ascii",
    vtk_precision: str = "float32",
    plate_carree_seam_mode: str = "wrap",
) -> int:
    """Write Natural Earth rivers as legacy VTK ``POLYDATA`` lines."""
    return write_natural_earth_lines_vtk(
        output_path=output_path,
        projection=projection,
        radius=radius,
        resolution=resolution,
        category="physical",
        dataset_name="rivers_lake_centerlines",
        feature_label="river",
        radius_offset=radius_offset,
        bbox=bbox,
        circle=circle,
        vtk_format=vtk_format,
        vtk_precision=vtk_precision,
        plate_carree_seam_mode=plate_carree_seam_mode,
    )


def write_country_vtk(
    output_path: Path,
    projection: str,
    radius: float,
    resolution: str,
    radius_offset: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    vtk_format: str = "ascii",
    vtk_precision: str = "float32",
    plate_carree_seam_mode: str = "wrap",
) -> int:
    """Write Natural Earth country boundaries as legacy VTK ``POLYDATA`` lines."""
    return write_natural_earth_lines_vtk(
        output_path=output_path,
        projection=projection,
        radius=radius,
        resolution=resolution,
        category="cultural",
        dataset_name="admin_0_boundary_lines_land",
        feature_label="country boundary",
        radius_offset=radius_offset,
        bbox=bbox,
        circle=circle,
        vtk_format=vtk_format,
        vtk_precision=vtk_precision,
        plate_carree_seam_mode=plate_carree_seam_mode,
    )


def write_province_vtk(
    output_path: Path,
    projection: str,
    radius: float,
    resolution: str,
    radius_offset: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    vtk_format: str = "ascii",
    vtk_precision: str = "float32",
    plate_carree_seam_mode: str = "wrap",
) -> int:
    """Write Natural Earth province boundaries as legacy VTK ``POLYDATA`` lines."""
    return write_natural_earth_lines_vtk(
        output_path=output_path,
        projection=projection,
        radius=radius,
        resolution=resolution,
        category="cultural",
        dataset_name="admin_1_states_provinces_lines",
        feature_label="province boundary",
        radius_offset=radius_offset,
        bbox=bbox,
        circle=circle,
        vtk_format=vtk_format,
        vtk_precision=vtk_precision,
        plate_carree_seam_mode=plate_carree_seam_mode,
    )


def build_axis_values(step: float, start: float, end: float) -> list[float]:
    """Generate regularly spaced axis values while keeping endpoints stable."""
    if step <= 0.0:
        raise ValueError("Graticule spacing must be positive")
    values = []
    current = start
    epsilon = step * 1e-9
    while current <= end + epsilon:
        rounded = round(current, 10)
        if rounded > end and rounded - end < epsilon:
            rounded = end
        values.append(rounded)
        current += step
    return values


def write_graticule_vtk(
    output_path: Path,
    projection: str,
    radius: float,
    spacing: tuple[float, float],
    radius_offset: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    vtk_format: str = "ascii",
    vtk_precision: str = "float32",
    plate_carree_seam_mode: str = "wrap",
) -> int:
    """Write longitude and latitude guide lines as VTK ``POLYDATA``."""
    dlon, dlat = spacing
    if dlon <= 0.0 or dlat <= 0.0:
        raise ValueError(
            "Graticule spacing must be positive in both longitude and latitude"
        )

    if bbox is not None and circle is not None:
        raise ValueError("Use either --bbox or --circle, not both")

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
    elif circle is not None:
        lon_min, lat_min, lon_max, lat_max = circle_to_bbox(circle, radius)
    else:
        lon_min, lat_min, lon_max, lat_max = -180.0, -90.0, 180.0, 90.0

    lat_values = build_axis_values(dlat, -90.0, 90.0)
    lon_values = build_axis_values(dlon, -180.0, 180.0)
    if bbox is not None:
        lat_values = [lat for lat in lat_values if lat_min <= lat <= lat_max]
        if lon_min <= lon_max:
            lon_values = [lon for lon in lon_values if lon_min <= lon <= lon_max]
        else:
            lon_values = [lon for lon in lon_values if lon >= lon_min or lon <= lon_max]

    all_points: list[np.ndarray] = []
    line_lengths: list[int] = []

    # Meridians: constant longitude, varying latitude.
    for lon in lon_values:
        lats = np.linspace(lat_min, lat_max, 181)
        lons = np.full_like(lats, lon)
        if circle is not None:
            mask = circle_contains(lons, lats, circle, radius)
            segments = [
                np.column_stack((lons[group], lats[group]))
                for group in split_true_runs(mask)
                if group.size >= 2
            ]
            if not segments:
                continue
        else:
            segments = [np.column_stack((lons, lats))]

        for segment in segments:
            projected = project_polyline(
                segment[:, 0],
                segment[:, 1],
                projection,
                radius,
                radius_offset,
                plate_carree_seam_mode,
            )
            all_points.append(projected)
            line_lengths.append(int(projected.shape[0]))

    lon_range_crosses_dateline = lon_min > lon_max
    # Parallels: constant latitude, varying longitude.
    for lat in lat_values:
        if lon_range_crosses_dateline:
            lon_segments = [
                np.linspace(lon_min, 180.0, 181),
                np.linspace(-180.0, lon_max, 181),
            ]
        else:
            lon_segments = [np.linspace(lon_min, lon_max, 361)]

        for lons in lon_segments:
            lats = np.full_like(lons, lat)
            if circle is not None:
                mask = circle_contains(lons, lats, circle, radius)
                segments = [
                    np.column_stack((lons[group], lats[group]))
                    for group in split_true_runs(mask)
                    if group.size >= 2
                ]
                if not segments:
                    continue
            else:
                segments = [np.column_stack((lons, lats))]

            for segment in segments:
                projected = project_polyline(
                    segment[:, 0],
                    segment[:, 1],
                    projection,
                    radius,
                    radius_offset,
                    plate_carree_seam_mode,
                )
                all_points.append(projected)
                line_lengths.append(int(projected.shape[0]))

    if not all_points:
        raise RuntimeError("No graticule lines were generated for the requested region")

    points = np.vstack(all_points)
    float_dtype, vtk_float_name = vtk_precision_spec(vtk_precision)
    with output_path.open("wb") as fh:
        write_header(
            fh,
            f"Longitude-latitude graticule [{dlon} x {dlat} deg]",
            "POLYDATA",
            vtk_format,
        )
        fh.write(f"POINTS {points.shape[0]} {vtk_float_name}\n".encode("ascii"))
        write_numeric_array(
            fh, np.asarray(points, dtype=float_dtype), vtk_format
        )
        total_size = sum(length + 1 for length in line_lengths)
        fh.write(f"LINES {len(line_lengths)} {total_size}\n".encode("ascii"))
        line_rows = []
        offset = 0
        for length in line_lengths:
            row = np.empty(length + 1, dtype=np.int32)
            row[0] = length
            row[1:] = np.arange(offset, offset + length, dtype=np.int32)
            line_rows.append(row)
            offset += length
        if vtk_format == "ascii":
            for row in line_rows:
                fh.write((" ".join(str(int(v)) for v in row) + "\n").encode("ascii"))
        else:
            line_blob = np.concatenate(line_rows)
            write_numeric_array(fh, line_blob, vtk_format)

    return len(line_lengths)


def main() -> int:
    """Command-line entry point."""
    total_start = time.perf_counter()
    args = parse_args()
    data_path = Path(args.data_file) if args.data_file is not None else None
    if args.list_variables:
        if data_path is None:
            print("Error: data_file is required with --list-variables", file=sys.stderr)
            return 1
        try:
            step_start = time.perf_counter()
            list_variables(data_path)
            log_message(
                f"Listed variables from {data_path} in "
                f"{format_duration(time.perf_counter() - step_start)}"
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    wants_field_export = args.variable is not None or args.output is not None
    wants_overlay = (
        args.coastline_output is not None
        or args.river_output is not None
        or args.country_output is not None
        or args.province_output is not None
        or args.graticule_output is not None
    )
    if wants_field_export and (
        data_path is None or args.grid_file is None or args.variable is None
    ):
        print(
            "Error: data_file, grid_file, and variable are required for field export",
            file=sys.stderr,
        )
        return 1
    if not wants_field_export and not wants_overlay:
        print(
            "Error: request either a field export (data_file grid_file variable) "
            "or at least one overlay output option. Use -h for help.",
            file=sys.stderr,
        )
        return 1
    if not wants_field_export and args.output is not None:
        print("Error: --output is only valid when exporting a field", file=sys.stderr)
        return 1

    grid_path = Path(args.grid_file) if args.grid_file is not None else None
    output_path = choose_output_path(args) if wants_field_export else None
    bbox = tuple(args.bbox) if args.bbox is not None else None
    circle = tuple(args.circle) if args.circle is not None else None

    try:
        time_indices = parse_index_list(args.time_index, "--time-index")
        level_indices = parse_index_list(args.level_index, "--level-index")
        if args.coarsen_level < 0:
            raise ValueError("Coarsen level must be non-negative")
        radius = resolve_radius(args.radius)
        cells = None
        field_exports: list[dict[str, object]] = []
        if wants_field_export:
            step_start = time.perf_counter()
            ensure_variable_exists(data_path, args.variable)
            log_message(
                f"Variable {args.variable}: found in {data_path} "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
            step_start = time.perf_counter()
            points, cells, parent_cell_index = read_mesh(grid_path, radius)
            log_message(
                f"Mesh: {cells.shape[0]} cells, {points.shape[0]} points "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
            is_batch = (len(time_indices) * len(level_indices)) > 1
            seen_output_paths: set[Path] = set()
            base_points = points
            base_cells = cells
            base_parent_cell_index = parent_cell_index
            for time_index in time_indices:
                for level_index in level_indices:
                    step_start = time.perf_counter()
                    values, units, title, used_time_index, used_level_index = (
                        read_field(
                            data_path,
                            args.variable,
                            time_index,
                            level_index,
                            expected_ncells=base_cells.shape[0],
                        )
                    )
                    log_message(
                        f"Field {args.variable}: {values.size} values for "
                        f"time-index={time_index}, level-index={level_index} "
                        f"({format_duration(time.perf_counter() - step_start)})"
                    )
                    step_start = time.perf_counter()
                    current_points = base_points
                    current_cells = base_cells
                    current_values = values
                    applied_coarsen_level = 0
                    if args.coarsen_level > 0:
                        (
                            current_points,
                            current_cells,
                            current_values,
                            applied_coarsen_level,
                        ) = coarsen_mesh(
                            current_points,
                            current_cells,
                            current_values,
                            base_parent_cell_index,
                            args.coarsen_level,
                        )
                    current_points, current_cells, current_values = subset_field_mesh(
                        current_points,
                        current_cells,
                        current_values,
                        radius,
                        bbox=bbox,
                        circle=circle,
                    )
                    current_points, current_cells = project_mesh(
                        current_points,
                        current_cells,
                        args.projection,
                        radius,
                        args.field_radius_offset,
                        args.plate_carree_seam_mode,
                    )
                    stats = summarize_values(current_values)
                    current_output_path = build_output_path(
                        output_path,
                        time_index if used_time_index else None,
                        level_index if used_level_index else None,
                        is_batch,
                    )
                    if current_output_path in seen_output_paths:
                        raise ValueError(
                            "Requested multiple indices for a dimension that is not "
                            f"used by variable {args.variable!r}; output path collision "
                            f"at {current_output_path}"
                        )
                    seen_output_paths.add(current_output_path)
                    log_message(
                        f"Processed: {current_cells.shape[0]} output cells for "
                        f"{current_output_path} "
                        f"({format_duration(time.perf_counter() - step_start)})"
                    )
                    step_start = time.perf_counter()
                    write_legacy_vtk(
                        current_output_path,
                        current_points,
                        current_cells,
                        current_values,
                        args.variable,
                        title,
                        units,
                        args.vtk_format,
                        args.vtk_precision,
                    )
                    log_message(
                        f"Field output: {current_output_path} "
                        f"({format_duration(time.perf_counter() - step_start)})"
                    )
                    field_exports.append(
                        {
                            "output_path": current_output_path,
                            "cells": current_cells.shape[0],
                            "time_index": time_index if used_time_index else None,
                            "level_index": level_index if used_level_index else None,
                            "stats": stats,
                            "applied_coarsen_level": applied_coarsen_level,
                        }
                    )
        coastline_count = None
        coastline_path = None
        river_count = None
        river_path = None
        country_count = None
        country_path = None
        province_count = None
        province_path = None
        graticule_count = None
        graticule_path = None
        if args.coastline_output:
            coastline_path = Path(args.coastline_output)
            step_start = time.perf_counter()
            coastline_count = write_coastline_vtk(
                coastline_path,
                projection=args.projection,
                radius=radius,
                resolution=args.coastline_resolution,
                radius_offset=args.coastline_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                vtk_precision=args.vtk_precision,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
            log_message(
                f"Coastlines: {coastline_path} "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
        if args.river_output:
            river_path = Path(args.river_output)
            step_start = time.perf_counter()
            river_count = write_river_vtk(
                river_path,
                projection=args.projection,
                radius=radius,
                resolution=args.river_resolution,
                radius_offset=args.river_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                vtk_precision=args.vtk_precision,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
            log_message(
                f"Rivers: {river_path} "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
        if args.country_output:
            country_path = Path(args.country_output)
            step_start = time.perf_counter()
            country_count = write_country_vtk(
                country_path,
                projection=args.projection,
                radius=radius,
                resolution=args.country_resolution,
                radius_offset=args.country_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                vtk_precision=args.vtk_precision,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
            log_message(
                f"Country boundaries: {country_path} "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
        if args.province_output:
            province_path = Path(args.province_output)
            step_start = time.perf_counter()
            province_count = write_province_vtk(
                province_path,
                projection=args.projection,
                radius=radius,
                resolution=args.province_resolution,
                radius_offset=args.province_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                vtk_precision=args.vtk_precision,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
            log_message(
                f"Province boundaries: {province_path} "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
        if args.graticule_output:
            graticule_path = Path(args.graticule_output)
            step_start = time.perf_counter()
            graticule_count = write_graticule_vtk(
                graticule_path,
                projection=args.projection,
                radius=radius,
                spacing=tuple(args.graticule_spacing),
                radius_offset=args.graticule_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                vtk_precision=args.vtk_precision,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
            log_message(
                f"Graticule: {graticule_path} "
                f"({format_duration(time.perf_counter() - step_start)})"
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if wants_field_export:
        for export in field_exports:
            index_parts = []
            if export["time_index"] is not None:
                index_parts.append(f"time-index={export['time_index']}")
            if export["level_index"] is not None:
                index_parts.append(f"level-index={export['level_index']}")
            index_text = f" ({', '.join(index_parts)})" if index_parts else ""
            print(
                f"Wrote {export['output_path']} with {export['cells']} cells "
                f"and variable {args.variable}{index_text}"
            )
    if args.projection != "sphere" or not wants_field_export:
        print(f"Projection: {args.projection}")
    if args.projection == "plate-carree":
        print(f"Plate-carree seam mode: {args.plate_carree_seam_mode}")
    if wants_field_export and args.coarsen_level != 0:
        for export in field_exports:
            index_parts = []
            if export["time_index"] is not None:
                index_parts.append(f"time-index={export['time_index']}")
            if export["level_index"] is not None:
                index_parts.append(f"level-index={export['level_index']}")
            index_text = f" ({', '.join(index_parts)})" if index_parts else ""
            print(
                f"Coarsen level{index_text}: requested={args.coarsen_level} "
                f"applied={export['applied_coarsen_level']}"
            )
    if wants_field_export and args.field_radius_offset != 0.0:
        print(f"Field radius offset: {args.field_radius_offset:.16g}")
    if field_exports:
        for export in field_exports:
            stats = export["stats"]
            index_parts = []
            if export["time_index"] is not None:
                index_parts.append(f"time-index={export['time_index']}")
            if export["level_index"] is not None:
                index_parts.append(f"level-index={export['level_index']}")
            index_text = f" ({', '.join(index_parts)})" if index_parts else ""
            print(
                f"Field stats{index_text}: "
                f"count={stats['count']} finite={stats['finite_count']} nan={stats['nan_count']} "
                f"min={stats['min']:.16g} max={stats['max']:.16g} mean={stats['mean']:.16g}"
            )
    elif wants_overlay:
        print(f"Radius: {radius:.16g}")
    if coastline_path is not None and coastline_count is not None:
        print(
            f"Wrote {coastline_path} with {coastline_count} coastline polylines "
            f"from Natural Earth {args.coastline_resolution} "
            f"(radius offset {args.coastline_radius_offset:.16g})"
        )
    if river_path is not None and river_count is not None:
        print(
            f"Wrote {river_path} with {river_count} river polylines "
            f"from Natural Earth {args.river_resolution} "
            f"(radius offset {args.river_radius_offset:.16g})"
        )
    if country_path is not None and country_count is not None:
        print(
            f"Wrote {country_path} with {country_count} country boundary polylines "
            f"from Natural Earth {args.country_resolution} "
            f"(radius offset {args.country_radius_offset:.16g})"
        )
    if province_path is not None and province_count is not None:
        print(
            f"Wrote {province_path} with {province_count} province boundary polylines "
            f"from Natural Earth {args.province_resolution} "
            f"(radius offset {args.province_radius_offset:.16g})"
        )
    if graticule_path is not None and graticule_count is not None:
        print(
            f"Wrote {graticule_path} with {graticule_count} graticule polylines "
            f"(spacing {args.graticule_spacing[0]:.16g} x {args.graticule_spacing[1]:.16g} deg, "
            f"radius offset {args.graticule_radius_offset:.16g})"
        )
    print(f"Total time: {format_duration(time.perf_counter() - total_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
