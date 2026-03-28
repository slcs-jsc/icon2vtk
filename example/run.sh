#!/usr/bin/env bash

set -euo pipefail

# Run this script from the example/ directory.
# It writes the generated VTK files back into the same directory.

echo "[1/2] Sphere example"
echo "3-D field, one time step, one model level, coarsened once, with overlays."
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
