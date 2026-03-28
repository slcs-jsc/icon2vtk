#!/usr/bin/env bash

set -euo pipefail

# Quick-start driver for the bundled ICON2VTK examples.
#
# Run this script from the example/ directory:
#   cd example
#   bash run.sh
#
# The sample ICON netCDF input files live in ../data/.
# All generated VTK output is written into this example/ directory so that the
# saved ParaView state files in this folder can load the results directly.
#
# The script generates two small example scenes:
# 1. A spherical view of one 3-D ICON field slice.
# 2. A plate-carree map view of one 2-D ICON field slice.

echo "[1/2] Sphere example"
echo "3-D field, one time step, one model level, coarsened once, with overlays."

# This example shows a typical "global sphere" workflow:
# - read the 3-D sample atmosphere file from ../data/
# - export air temperature ("ta")
# - pick one time step and one vertical model level
# - coarsen the ICON mesh by one refinement level for a lighter output
# - write separate coastline and graticule overlays
# - lift the field slightly above the sphere so overlays are easier to see
python3 ../icon2vtk.py \
  ../data/aes_amip_atm_3d_qp_ml_19790101T000000Z.nc \
  ../data/icon_grid_0049_R02B04_G.nc \
  ta \
  --time-index 1 \
  --level-index 45 \
  --coarsen-level 1 \
  --field-radius-offset 5000 \
  --coastline-output sphere_coastlines.vtk \
  --graticule-output sphere_graticule.vtk \
  -o sphere_ta_t1_l45.vtk

echo
echo "[2/2] Plate-carree example"
echo "2-D field, flat map projection, clipped seam handling, with overlays."

# This example shows a "flat map" workflow:
# - read the 2-D sample surface field file from ../data/
# - export surface temperature ("ts")
# - project the ICON mesh to a plate-carree longitude/latitude map
# - clip cells at the map seam to avoid wrap-around artifacts
# - write separate coastline and graticule overlays for the same projection
python3 ../icon2vtk.py \
  ../data/aes_amip_atm_2d_P1D_ml_19790101T000000Z.nc \
  ../data/icon_grid_0049_R02B04_G.nc \
  ts \
  --projection plate-carree \
  --plate-carree-seam-mode clip \
  --time-index 1 \
  --coastline-output plate_carree_coastlines.vtk \
  --graticule-output plate_carree_graticule.vtk \
  -o plate_carree_ts_bbox.vtk

echo
echo "Done. Load the generated VTK files from this example/ directory in ParaView."
echo "Tip: the saved ParaView states sphere.pvsm and plate_carree.pvsm match these outputs."
