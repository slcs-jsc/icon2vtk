#!/usr/bin/env bash

set -euo pipefail

# This runner is intended to be executed from tests/.
PYTHON_BIN="${PYTHON_BIN:-python3}"
UPDATE_REFS=0

PASS=0
FAIL=0
SKIP=0

log()  { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "✅ $*"; PASS=$((PASS+1)); }
bad()  { echo "❌ $*"; FAIL=$((FAIL+1)); }
skip() { echo "⚠️  $*"; SKIP=$((SKIP+1)); }

usage() {
  cat <<'EOF'
Usage: ./run.sh [--update_refs]

Options:
  --update_refs  Replace tests/data.ref with the newly generated test outputs
                 after a successful test run.
  -h, --help     Show this help text.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --update_refs)
        UPDATE_REFS=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done
}

# Remove the outputs owned by one case so stale files cannot hide regressions
# or get copied into the golden set by `--update_refs`.
reset_case_outputs() {
  local path
  for path in "$@"; do
    rm -f "$path"
  done
}

# Fail if the expected artifact was not created or is empty.
assert_file_exists_nonempty() {
  local f="$1"
  if [[ ! -s "$f" ]]; then
    bad "File missing or empty: $f"
    return 1
  fi
}

# Compare against the golden output. `cmp` handles byte-for-byte matches fast;
# `diff` gives a robust fallback for text VTK files without changing behavior.
compare_exact_or_diff() {
  local got="$1"
  local ref="$2"
  if [[ ! -f "$ref" ]]; then
    bad "Reference file missing: $ref"
    return 1
  fi
  if cmp -s "$got" "$ref"; then
    return 0
  fi
  if diff -u "$ref" "$got" > /dev/null; then
    return 0
  fi
  bad "Files differ: $got != $ref"
  return 1
}

# Promote the just-generated artifacts to golden files, but only after the
# whole suite has passed so broken outputs cannot be recorded accidentally.
update_reference_data() {
  rm -rf "data.ref"
  mkdir -p "data.ref"
  cp -r "data/." "data.ref/"
  log "Reference data updated: tests/data.ref"
}

# 2D regression: export one time slice with the plate-carree projection,
# clip the seam, and also emit a graticule companion file.
run_case_core_2d_plate_carree() {
  local case_name="core_2d_plate_carree"
  local out="data/core_2d_plate_carree_ts_t1_clip.vtk"
  local grat="data/core_2d_plate_carree_graticule.vtk"

  reset_case_outputs "$out" "$grat"

  log "Case: $case_name"
  "$PYTHON_BIN" ../icon2vtk.py \
    ../data/aes_amip_atm_2d_P1D_ml_19790101T000000Z.nc \
    ../data/icon_grid_0049_R02B04_G.nc \
    ts \
    --projection plate-carree \
    --plate-carree-seam-mode clip \
    --time-index 1 \
    --graticule-output "$grat" \
    -o "$out"

  assert_file_exists_nonempty "$out" || return 1
  assert_file_exists_nonempty "$grat" || return 1

  compare_exact_or_diff "$out"  "data.ref/core_2d_plate_carree_ts_t1_clip.vtk" || return 1
  compare_exact_or_diff "$grat" "data.ref/core_2d_plate_carree_graticule.vtk" || return 1

  ok "$case_name"
}

# ASCII regression: exercise the non-default legacy VTK encoding path.
run_case_core_2d_ascii_vtk() {
  local case_name="core_2d_ascii_vtk"
  local out="data/core_2d_ascii_vtk_ts_t1_ascii.vtk"

  reset_case_outputs "$out"

  log "Case: $case_name"
  "$PYTHON_BIN" ../icon2vtk.py \
    ../data/aes_amip_atm_2d_P1D_ml_19790101T000000Z.nc \
    ../data/icon_grid_0049_R02B04_G.nc \
    ts \
    --projection sphere \
    --time-index 1 \
    --vtk-format ascii \
    -o "$out"

  assert_file_exists_nonempty "$out" || return 1
  compare_exact_or_diff "$out" "data.ref/core_2d_ascii_vtk_ts_t1_ascii.vtk" || return 1

  ok "$case_name"
}

# 3D regression: export a single vertical level on the sphere with coarsening
# enabled and a radial offset so the field is rendered above the globe.
run_case_core_3d_sphere_coarsen() {
  local case_name="core_3d_sphere_coarsen"
  local out="data/core_3d_sphere_coarsen_ta_t1_l45_c1.vtk"
  local grat="data/core_3d_sphere_coarsen_graticule.vtk"

  reset_case_outputs "$out" "$grat"

  log "Case: $case_name"
  "$PYTHON_BIN" ../icon2vtk.py \
    ../data/aes_amip_atm_3d_qp_ml_19790101T000000Z.nc \
    ../data/icon_grid_0049_R02B04_G.nc \
    ta \
    --time-index 1 \
    --level-index 45 \
    --coarsen-level 1 \
    --field-radius-offset 5000 \
    --graticule-output "$grat" \
    -o "$out"

  assert_file_exists_nonempty "$out" || return 1
  assert_file_exists_nonempty "$grat" || return 1

  compare_exact_or_diff "$out"  "data.ref/core_3d_sphere_coarsen_ta_t1_l45_c1.vtk" || return 1
  compare_exact_or_diff "$grat" "data.ref/core_3d_sphere_coarsen_graticule.vtk" || return 1

  ok "$case_name"
}

# Natural Earth overlay regression. This runs by default because the line
# overlays are expected to be available together; set
# RUN_NATURAL_EARTH_TESTS=0 to disable them.
run_case_overlay_natural_earth() {
  local case_name="overlay_natural_earth"
  local coast="data/overlay_coastline.vtk"
  local river="data/overlay_river.vtk"
  local country="data/overlay_country.vtk"
  local province="data/overlay_province.vtk"

  reset_case_outputs "$coast" "$river" "$country" "$province"

  if [[ "${RUN_NATURAL_EARTH_TESTS:-1}" == "0" ]]; then
    skip "$case_name (RUN_NATURAL_EARTH_TESTS=0)"
    return 0
  fi

  log "Case: $case_name"
  if ! "$PYTHON_BIN" ../icon2vtk.py \
      --projection sphere \
      --coastline-output "$coast" \
      --river-output "$river" \
      --country-output "$country" \
      --province-output "$province"; then
    skip "$case_name (Natural Earth overlays unavailable or offline)"
    return 0
  fi

  assert_file_exists_nonempty "$coast" || return 1
  assert_file_exists_nonempty "$river" || return 1
  assert_file_exists_nonempty "$country" || return 1
  assert_file_exists_nonempty "$province" || return 1

  compare_exact_or_diff "$coast" "data.ref/overlay_coastline.vtk" || return 1
  compare_exact_or_diff "$river" "data.ref/overlay_river.vtk" || return 1
  compare_exact_or_diff "$country" "data.ref/overlay_country.vtk" || return 1
  compare_exact_or_diff "$province" "data.ref/overlay_province.vtk" || return 1

  ok "$case_name"
}

main() {
  local case_runner

  parse_args "$@"

  # Keep all generated artifacts under tests/data/.
  mkdir -p "data"

  for case_runner in \
    run_case_core_2d_plate_carree \
    run_case_core_2d_ascii_vtk \
    run_case_core_3d_sphere_coarsen \
    run_case_overlay_natural_earth
  do
    "$case_runner" || true
  done

  echo
  echo "==== Results ===="
  echo "PASS: $PASS"
  echo "FAIL: $FAIL"
  echo "SKIP: $SKIP"

  if [[ "$FAIL" -gt 0 ]]; then
    exit 1
  fi

  if [[ "$UPDATE_REFS" -eq 1 ]]; then
    update_reference_data
  fi
}

main "$@"
