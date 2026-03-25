#!/usr/bin/env python3
"""Convert a cell-based 2-D ICON netCDF field to ASCII legacy VTK."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one cell-based field from an ICON netCDF file and write an "
            "legacy VTK unstructured grid using an ICON grid file."
        )
    )
    parser.add_argument("data_file", help="netCDF file holding the field to export")
    parser.add_argument(
        "grid_file",
        nargs="?",
        help="ICON grid file, e.g. icon_grid_*.nc",
    )
    parser.add_argument(
        "variable",
        nargs="?",
        help="Variable name to export",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output VTK file path (default: <variable>.vtk)",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Index to use for the time dimension if present (default: 0)",
    )
    parser.add_argument(
        "--level-index",
        type=int,
        default=0,
        help=(
            "Index to use for any extra non-cell dimension besides time "
            "(default: 0)"
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
            "Scale unit-sphere vertex coordinates to this radius. "
            "Defaults to the grid attribute sphere_radius when available."
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
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help=(
            "Restrict output to a lon/lat box in degrees. Cells are selected "
            "by ICON cell center; coastlines are clipped to the same box."
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
            "coastlines are filtered to the same region."
        ),
    )
    parser.add_argument(
        "--graticule-output",
        help=(
            "Optional VTK output path for longitude/latitude grid lines "
            "drawn on the sphere."
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
        default=20000.0,
        help=(
            "Add this offset to the exported graticule geometry. For sphere "
            "this increases the sphere radius; for flat projections it "
            "becomes a constant z offset (default: 20000)."
        ),
    )
    parser.add_argument(
        "--vtk-format",
        choices=("ascii", "binary"),
        default="binary",
        help="Legacy VTK output format (default: binary)",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List variables in the netCDF data file and exit",
    )
    return parser.parse_args()


def choose_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    return Path(f"{args.variable}.vtk")


def bbox_contains(lon_deg: np.ndarray, lat_deg: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mask = (lat_deg >= lat_min) & (lat_deg <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (lon_deg >= lon_min) & (lon_deg <= lon_max)
    else:
        lon_mask = (lon_deg >= lon_min) | (lon_deg <= lon_max)
    return lon_mask & lat_mask


def normalize_lon(lon_deg: np.ndarray) -> np.ndarray:
    return ((lon_deg + 180.0) % 360.0) - 180.0


def great_circle_distance_km(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    center_lon_deg: float,
    center_lat_deg: float,
    radius_m: float,
) -> np.ndarray:
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


def subset_mesh(points: np.ndarray, cells: np.ndarray, cell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_cells = cells[cell_mask]
    if selected_cells.size == 0:
        raise ValueError("No ICON cells fall inside the requested bounding box")

    used_vertices, inverse = np.unique(selected_cells.ravel(), return_inverse=True)
    subset_points = points[used_vertices]
    subset_cells = inverse.reshape(selected_cells.shape)
    return subset_points, subset_cells


def compact_cells(points: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cells.size == 0:
        raise ValueError("No ICON cells remain after coarsening")
    used_vertices, inverse = np.unique(cells.ravel(), return_inverse=True)
    return points[used_vertices], inverse.reshape(cells.shape)


def average_values(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite))


def order_triangle_vertices(points: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    triangle_points = points[vertex_ids]
    centroid = np.mean(triangle_points, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0.0:
        return vertex_ids

    normal = centroid / centroid_norm
    first_vec = triangle_points[0] - centroid
    first_norm = np.linalg.norm(first_vec)
    if first_norm == 0.0:
        return vertex_ids
    axis_x = first_vec / first_norm
    axis_y = np.cross(normal, axis_x)
    axis_y_norm = np.linalg.norm(axis_y)
    if axis_y_norm == 0.0:
        return vertex_ids
    axis_y /= axis_y_norm

    rel = triangle_points - centroid
    angles = np.arctan2(rel @ axis_y, rel @ axis_x)
    ordered = vertex_ids[np.argsort(angles)]

    ordered_points = points[ordered]
    orientation = np.dot(
        np.cross(ordered_points[1] - ordered_points[0], ordered_points[2] - ordered_points[0]),
        centroid,
    )
    if orientation < 0.0:
        ordered[[1, 2]] = ordered[[2, 1]]
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
    if parent_ids.shape[0] != cells.shape[0]:
        raise ValueError("parent id array length does not match the selected cells")
    if cell_ids.shape[0] != cells.shape[0] or cell_levels.shape[0] != cells.shape[0]:
        raise ValueError("cell metadata length does not match the selected cells")

    families: dict[int, list[int]] = {}
    candidate_mask = cell_levels == candidate_level
    for cell_idx in np.flatnonzero(candidate_mask):
        parent_idx = parent_ids[cell_idx]
        parent_id = int(parent_idx)
        if parent_id <= 0:
            continue
        families.setdefault(parent_id, []).append(cell_idx)

    coarsened_cells: list[np.ndarray] = []
    coarsened_values: list[float] = []
    coarsened_ids: list[int] = []
    coarsened_levels: list[int] = []
    keep_child_mask = np.ones(cells.shape[0], dtype=bool)

    for sibling_indices in families.values():
        if len(sibling_indices) != 4:
            continue

        sibling_cells = cells[sibling_indices]
        vertex_ids, counts = np.unique(sibling_cells.ravel(), return_counts=True)
        parent_vertices = vertex_ids[counts == 1]
        if parent_vertices.size != 3:
            continue

        ordered_vertices = order_triangle_vertices(points, parent_vertices.astype(np.int64))
        coarsened_cells.append(ordered_vertices)
        coarsened_values.append(average_values(values[np.asarray(sibling_indices, dtype=np.int64)]))
        coarsened_ids.append(int(parent_ids[sibling_indices[0]]))
        coarsened_levels.append(candidate_level + 1)
        keep_child_mask[np.asarray(sibling_indices, dtype=np.int64)] = False

    remaining_cells = cells[keep_child_mask]
    remaining_values = values[keep_child_mask]
    remaining_ids = cell_ids[keep_child_mask]
    remaining_levels = cell_levels[keep_child_mask]

    if coarsened_cells:
        combined_cells = np.vstack((remaining_cells, np.asarray(coarsened_cells, dtype=np.int64)))
        combined_values = np.concatenate(
            (remaining_values, np.asarray(coarsened_values, dtype=np.float64))
        )
        combined_ids = np.concatenate((remaining_ids, np.asarray(coarsened_ids, dtype=np.int64)))
        combined_levels = np.concatenate(
            (remaining_levels, np.asarray(coarsened_levels, dtype=np.int64))
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
        bool(coarsened_cells),
    )


def coarsen_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    parent_cell_index: np.ndarray | None,
    coarsen_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
        current_parent_ids = ((current_cell_ids - 1) // 4) + 1

    return current_points, current_cells, current_values, applied_levels


def read_mesh(
    grid_path: Path,
    radius_override: float | None,
    bbox: tuple[float, float, float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with Dataset(grid_path) as ds:
        required = [
            "vertex_of_cell",
            "cartesian_x_vertices",
            "cartesian_y_vertices",
            "cartesian_z_vertices",
        ]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"Grid file is missing required variables: {', '.join(missing)}")

        radius = radius_override
        if radius is None:
            radius = float(getattr(ds, "sphere_radius", 1.0))

        x = np.asarray(ds.variables["cartesian_x_vertices"][:], dtype=np.float64) * radius
        y = np.asarray(ds.variables["cartesian_y_vertices"][:], dtype=np.float64) * radius
        z = np.asarray(ds.variables["cartesian_z_vertices"][:], dtype=np.float64) * radius
        points = np.column_stack((x, y, z))

        connectivity = np.asarray(ds.variables["vertex_of_cell"][:], dtype=np.int64)
        if connectivity.ndim != 2 or connectivity.shape[0] != 3:
            raise ValueError(
                "Expected vertex_of_cell to have shape (3, ncells) for triangular ICON cells"
            )

        cells = connectivity.T - 1
        if np.any(cells < 0):
            raise ValueError("vertex_of_cell appears not to be 1-based as expected")
        parent_cell_index = None
        if "parent_cell_index" in ds.variables:
            parent_cell_index = np.asarray(ds.variables["parent_cell_index"][:], dtype=np.int64)
            if parent_cell_index.shape != (cells.shape[0],):
                raise ValueError(
                    "Expected parent_cell_index to have one entry per ICON cell"
                )

        if bbox is not None and circle is not None:
            raise ValueError("Use either --bbox or --circle, not both")

        if bbox is None and circle is None:
            cell_mask = np.ones(cells.shape[0], dtype=bool)
        else:
            if "clon" not in ds.variables or "clat" not in ds.variables:
                raise ValueError("Grid file is missing clon/clat needed for region selection")
            clon = np.rad2deg(np.asarray(ds.variables["clon"][:], dtype=np.float64))
            clat = np.rad2deg(np.asarray(ds.variables["clat"][:], dtype=np.float64))
            if bbox is not None:
                cell_mask = bbox_contains(clon, clat, bbox)
            else:
                cell_mask = circle_contains(clon, clat, circle, radius)

    if bbox is not None or circle is not None:
        points, cells = subset_mesh(points, cells, cell_mask)
        if parent_cell_index is not None:
            parent_cell_index = parent_cell_index[cell_mask]

    return points, cells, cell_mask, parent_cell_index


def read_radius(grid_path: Path, radius_override: float | None) -> float:
    if radius_override is not None:
        return float(radius_override)
    with Dataset(grid_path) as ds:
        return float(getattr(ds, "sphere_radius", 1.0))


def read_field(
    data_path: Path,
    variable_name: str,
    time_index: int,
    level_index: int,
    expected_ncells: int,
) -> tuple[np.ndarray, str | None, str]:
    with Dataset(data_path) as ds:
        if variable_name not in ds.variables:
            available = ", ".join(ds.variables.keys())
            raise ValueError(f"Variable {variable_name!r} not found. Available variables: {available}")

        var = ds.variables[variable_name]
        dims = var.dimensions
        shape = var.shape
        if "ncells" not in dims:
            raise ValueError(
                f"Variable {variable_name!r} is not cell-based; dimensions are {dims}"
            )

        selection: list[int | slice] = []
        used_extra_dim = False
        for dim_name, dim_size in zip(dims, shape):
            if dim_name == "ncells":
                selection.append(slice(None))
            elif dim_name == "time":
                if not 0 <= time_index < dim_size:
                    raise IndexError(
                        f"time index {time_index} out of range for size {dim_size}"
                    )
                selection.append(time_index)
            else:
                idx = 0 if dim_size == 1 else level_index
                if not 0 <= idx < dim_size:
                    raise IndexError(
                        f"level index {level_index} out of range for dimension "
                        f"{dim_name!r} of size {dim_size}"
                    )
                selection.append(idx)
                used_extra_dim = True

        data = np.ma.asarray(var[tuple(selection)])
        values = np.ma.filled(data, np.nan).astype(np.float64, copy=False)
        values = np.ravel(values)

        if values.size != expected_ncells:
            raise ValueError(
                f"Selected data has {values.size} values, expected {expected_ncells} cells"
            )

        units = getattr(var, "units", None)
        long_name = getattr(var, "long_name", variable_name)
        if used_extra_dim:
            long_name = f"{long_name} (level-index={level_index})"
        return values, units, long_name


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
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
    return "(" + ", ".join(str(v) for v in shape) + ")"


def list_variables(data_path: Path) -> None:
    with Dataset(data_path) as ds:
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
            long_text = long_name if long_name and long_name != "-" else name
            std_text = (
                f"; standard_name={standard_name}"
                if standard_name and standard_name != "-"
                else ""
            )
            unit_text = f"; units=[{units}]" if units and units != "-" else ""
            print(f"- {name}: {long_text}{std_text}{unit_text}")
            print(f"    dims={dims}; shape={shape}; grid={grid_type}; dtype={dtype}")
            print()


def write_header(fh, title: str, dataset_type: str, vtk_format: str) -> None:
    fh.write(b"# vtk DataFile Version 3.0\n")
    fh.write(title[:255].encode("ascii", errors="replace") + b"\n")
    fh.write(vtk_format.upper().encode("ascii") + b"\n")
    fh.write(f"DATASET {dataset_type}\n".encode("ascii"))


def write_numeric_array(fh, array: np.ndarray, vtk_format: str, fmt: str) -> None:
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
) -> None:
    npoints = points.shape[0]
    ncells = cells.shape[0]

    with output_path.open("wb") as fh:
        header = title
        if units:
            header = f"{header} [{units}]"
        write_header(fh, header, "UNSTRUCTURED_GRID", vtk_format)

        fh.write(f"POINTS {npoints} double\n".encode("ascii"))
        write_numeric_array(fh, np.asarray(points, dtype=np.float64), vtk_format, "double")

        fh.write(f"CELLS {ncells} {ncells * 4}\n".encode("ascii"))
        cell_array = np.column_stack(
            (
                np.full(ncells, 3, dtype=np.int32),
                np.asarray(cells, dtype=np.int32),
            )
        )
        write_numeric_array(fh, cell_array, vtk_format, "int")

        fh.write(f"CELL_TYPES {ncells}\n".encode("ascii"))
        write_numeric_array(fh, np.full(ncells, 5, dtype=np.int32), vtk_format, "int")

        fh.write(f"CELL_DATA {ncells}\n".encode("ascii"))
        fh.write(f"SCALARS {sanitize_name(variable_name)} double 1\n".encode("ascii"))
        fh.write(b"LOOKUP_TABLE default\n")
        write_numeric_array(fh, np.asarray(values, dtype=np.float64), vtk_format, "double")


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    return cleaned or "field"


def default_coastline_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_coastlines.vtk")


def xyz_to_lonlat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray, radius: float) -> np.ndarray:
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
    if projection == "sphere":
        return lonlat_to_xyz(lon_deg, lat_deg, radius + offset)
    if projection == "plate-carree":
        x = radius * np.deg2rad(lon_deg)
        y = radius * np.deg2rad(lat_deg)
        z = np.full_like(x, offset, dtype=np.float64)
        return np.column_stack((x, y, z))
    raise ValueError(f"Unsupported projection {projection!r}")


def project_xyz(points: np.ndarray, projection: str, radius: float, offset: float = 0.0) -> np.ndarray:
    if projection == "sphere":
        if offset == 0.0:
            return points
        if radius == 0.0:
            raise ValueError("Cannot apply a sphere offset when radius is zero")
        scale = (radius + offset) / radius
        return points * scale
    lon_deg, lat_deg = xyz_to_lonlat(points)
    return project_lonlat(lon_deg, lat_deg, projection, radius, offset)


def unwrap_longitudes_for_cell(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    non_polar_mask = np.abs(lat_deg) < 89.999999
    if np.any(non_polar_mask):
        reference_lon = float(lon_deg[np.flatnonzero(non_polar_mask)[0]])
    else:
        reference_lon = float(lon_deg[0])

    adjusted_lon = reference_lon + ((lon_deg - reference_lon + 180.0) % 360.0) - 180.0
    polar_mask = ~non_polar_mask
    if np.any(polar_mask) and np.any(non_polar_mask):
        adjusted_lon = adjusted_lon.copy()
        adjusted_lon[polar_mask] = float(np.mean(adjusted_lon[non_polar_mask]))
    return adjusted_lon


def wrap_longitudes_to_primary_range(lon_deg: np.ndarray) -> np.ndarray:
    wrapped_lon = lon_deg.copy()
    center_lon = float(np.mean(wrapped_lon))
    wrapped_lon -= 360.0 * math.floor((center_lon + 180.0) / 360.0)
    if np.mean(wrapped_lon) > 180.0:
        wrapped_lon -= 360.0
    elif np.mean(wrapped_lon) < -180.0:
        wrapped_lon += 360.0
    return wrapped_lon


def clip_longitudes_to_primary_range(lon_deg: np.ndarray) -> np.ndarray:
    return np.clip(lon_deg, -180.0, 180.0)


def unwrap_longitudes_for_polyline(lon_deg: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(lon_deg)))


def project_polyline(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    projection: str,
    radius: float,
    offset: float = 0.0,
    plate_carree_seam_mode: str = "wrap",
) -> np.ndarray:
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
    if projection == "sphere":
        return project_xyz(points, projection, radius, offset), cells

    if projection != "plate-carree":
        raise ValueError(f"Unsupported projection {projection!r}")

    lon_deg, lat_deg = xyz_to_lonlat(points)
    projected_cells: list[np.ndarray] = []
    projected_points: list[np.ndarray] = []

    for cell in cells:
        cell_lat = lat_deg[cell]
        cell_lon = unwrap_longitudes_for_cell(lon_deg[cell], cell_lat)
        cell_lon = wrap_longitudes_to_primary_range(cell_lon)
        if plate_carree_seam_mode == "clip":
            cell_lon = clip_longitudes_to_primary_range(cell_lon)
        elif plate_carree_seam_mode != "wrap":
            raise ValueError(
                f"Unsupported plate-carree seam mode {plate_carree_seam_mode!r}"
            )
        cell_points = project_lonlat(cell_lon, cell_lat, projection, radius, offset)
        start = len(projected_points)
        projected_points.extend(cell_points)
        projected_cells.append(np.arange(start, start + len(cell), dtype=np.int64))

    return np.asarray(projected_points, dtype=np.float64), np.asarray(projected_cells, dtype=np.int64)


def iter_linestring_coords(geometry) -> list[np.ndarray]:
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
    plate_carree_seam_mode: str = "wrap",
) -> int:
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import GeometryCollection, box
    except ImportError as exc:
        raise RuntimeError(
            "Cartopy and Shapely are required for coastline export but are not available"
        ) from exc

    shapefile_path = shpreader.natural_earth(
        resolution=resolution,
        category="physical",
        name="coastline",
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
                [box(lon_min, lat_min, 180.0, lat_max), box(-180.0, lat_min, lon_max, lat_max)]
            )
    elif circle is not None:
        lon_min, lat_min, lon_max, lat_max = circle_to_bbox(circle, radius)
        if lon_min <= lon_max:
            bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
        else:
            bbox_geom = GeometryCollection(
                [box(lon_min, lat_min, 180.0, lat_max), box(-180.0, lat_min, lon_max, lat_max)]
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
                point_mask = circle_contains(coords[:, 0], coords[:, 1], circle, radius)
                filtered_segments = [
                    coords[group] for group in split_true_runs(point_mask) if group.size >= 2
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
        raise RuntimeError("No coastline geometries were found in the Cartopy dataset")

    points = np.vstack(all_points)
    with output_path.open("wb") as fh:
        write_header(fh, f"Natural Earth coastlines [{resolution}]", "POLYDATA", vtk_format)
        fh.write(f"POINTS {points.shape[0]} double\n".encode("ascii"))
        write_numeric_array(fh, np.asarray(points, dtype=np.float64), vtk_format, "double")
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
            write_numeric_array(fh, line_blob, vtk_format, "int")

    return len(line_lengths)


def build_axis_values(step: float, start: float, end: float) -> list[float]:
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
    plate_carree_seam_mode: str = "wrap",
) -> int:
    dlon, dlat = spacing
    if dlon <= 0.0 or dlat <= 0.0:
        raise ValueError("Graticule spacing must be positive in both longitude and latitude")

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

    all_points: list[np.ndarray] = []
    line_lengths: list[int] = []

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
    with output_path.open("wb") as fh:
        write_header(fh, f"Longitude-latitude graticule [{dlon} x {dlat} deg]", "POLYDATA", vtk_format)
        fh.write(f"POINTS {points.shape[0]} double\n".encode("ascii"))
        write_numeric_array(fh, np.asarray(points, dtype=np.float64), vtk_format, "double")
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
            write_numeric_array(fh, line_blob, vtk_format, "int")

    return len(line_lengths)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_file)
    if args.list_variables:
        try:
            list_variables(data_path)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.grid_file is None or args.variable is None:
        print(
            "Error: grid_file and variable are required unless --list-variables is used",
            file=sys.stderr,
        )
        return 1

    grid_path = Path(args.grid_file)
    output_path = choose_output_path(args)
    bbox = tuple(args.bbox) if args.bbox is not None else None
    circle = tuple(args.circle) if args.circle is not None else None

    try:
        if args.coarsen_level < 0:
            raise ValueError("Coarsen level must be non-negative")
        radius = read_radius(grid_path, args.radius)
        points, cells, cell_mask, parent_cell_index = read_mesh(
            grid_path,
            radius,
            bbox=bbox,
            circle=circle,
        )
        values, units, title = read_field(
            data_path,
            args.variable,
            args.time_index,
            args.level_index,
            expected_ncells=cell_mask.size,
        )
        values = values[cell_mask]
        applied_coarsen_level = 0
        if args.coarsen_level > 0:
            points, cells, values, applied_coarsen_level = coarsen_mesh(
                points,
                cells,
                values,
                parent_cell_index,
                args.coarsen_level,
            )
        points, cells = project_mesh(
            points,
            cells,
            args.projection,
            radius,
            args.field_radius_offset,
            args.plate_carree_seam_mode,
        )
        stats = summarize_values(values)
        write_legacy_vtk(
            output_path,
            points,
            cells,
            values,
            args.variable,
            title,
            units,
            args.vtk_format,
        )
        coastline_count = None
        coastline_path = None
        graticule_count = None
        graticule_path = None
        if args.coastline_output:
            coastline_path = Path(args.coastline_output)
            coastline_count = write_coastline_vtk(
                coastline_path,
                projection=args.projection,
                radius=radius,
                resolution=args.coastline_resolution,
                radius_offset=args.coastline_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
        if args.graticule_output:
            graticule_path = Path(args.graticule_output)
            graticule_count = write_graticule_vtk(
                graticule_path,
                projection=args.projection,
                radius=radius,
                spacing=tuple(args.graticule_spacing),
                radius_offset=args.graticule_radius_offset,
                bbox=bbox,
                circle=circle,
                vtk_format=args.vtk_format,
                plate_carree_seam_mode=args.plate_carree_seam_mode,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path} with {cells.shape[0]} cells and variable {args.variable}")
    if args.projection != "sphere":
        print(f"Projection: {args.projection}")
    if args.projection == "plate-carree":
        print(f"Plate-carree seam mode: {args.plate_carree_seam_mode}")
    if args.coarsen_level != 0:
        print(f"Coarsen level: requested={args.coarsen_level} applied={applied_coarsen_level}")
    if args.field_radius_offset != 0.0:
        print(f"Field radius offset: {args.field_radius_offset:.16g}")
    print(
        "Field stats: "
        f"count={stats['count']} finite={stats['finite_count']} nan={stats['nan_count']} "
        f"min={stats['min']:.16g} max={stats['max']:.16g} mean={stats['mean']:.16g}"
    )
    if coastline_path is not None and coastline_count is not None:
        print(
            f"Wrote {coastline_path} with {coastline_count} coastline polylines "
            f"from Natural Earth {args.coastline_resolution} "
            f"(radius offset {args.coastline_radius_offset:.16g})"
        )
    if graticule_path is not None and graticule_count is not None:
        print(
            f"Wrote {graticule_path} with {graticule_count} graticule polylines "
            f"(spacing {args.graticule_spacing[0]:.16g} x {args.graticule_spacing[1]:.16g} deg, "
            f"radius offset {args.graticule_radius_offset:.16g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
