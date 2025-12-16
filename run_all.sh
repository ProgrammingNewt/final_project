#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="sample_input/hw_4_1"
OUT_DIR="molecules_bandgaps_and_dipoles"

SUMMARY_FILE="${OUT_DIR}/summary.txt"
FULL_DIR="${OUT_DIR}/full_molecule_output"

./clean.sh
./build.sh

EXE="$(find build -type f -executable \( -name 'hw_4_1' -o -name 'hw_4_1*' \) | head -n 1 || true)"
if [[ -z "${EXE}" ]]; then
  echo "ERROR: Could not find executable in ./build/" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$FULL_DIR"
: > "$SUMMARY_FILE"

shopt -s nullglob
for cfg in "$INPUT_DIR"/*.json; do
  name="$(basename "$cfg" .json)"

  MOL_DIR="${FULL_DIR}/${name}"
  mkdir -p "$MOL_DIR"

  RAW_OUT="${MOL_DIR}/stdout.txt"
  CLEAN_TMP="$(mktemp)"

  "$EXE" "$cfg" > "$RAW_OUT"
  tr -d '\r' < "$RAW_OUT" > "$CLEAN_TMP"


  {
    echo
    echo "======================================================================"
    echo "======================================================================"
    echo " Molecule: ${name}"
    echo " Config file: ${cfg}"
    echo "======================================================================"
    echo

    echo ">>> FINAL MO ENERGIES (eV)"
    echo "---------------------------------------------------------------------"
    awk '
      /Final MO energies \(eV\)/ {mo=1; next}
      mo && /HOMO\/LUMO energies \(eV\)/ {exit}
      mo {print}
    ' "$CLEAN_TMP"
    echo

    echo ">>> HOMO / LUMO SUMMARY"
    echo "---------------------------------------------------------------------"
    awk '
      /HOMO_alpha/ ||
      /LUMO_alpha/ ||
      /HOMO_beta/  ||
      /LUMO_beta/  ||
      /Overall HOMO/ ||
      /Overall LUMO/ ||
      /HOMO-LUMO gap:/ {print}
    ' "$CLEAN_TMP"
    echo

    echo ">>> DIPOLE MATRICES (AO BASIS)"
    echo "---------------------------------------------------------------------"
    awk '
      /Dipole X matrix:/ {grab=1}
      grab && /Computed electron count/ {exit}
      grab {print}
    ' "$CLEAN_TMP"
    echo

    echo ">>> DIPOLE MOMENT (SUMMARY)"
    echo "---------------------------------------------------------------------"
    awk '
      /Electronic dipole \(au\):/ ||
      /Nuclear dipole/ ||
      /Total dipole/ ||
      /Dipole magnitude:/ {print}
    ' "$CLEAN_TMP"

    echo
    echo "======================================================================"
    echo
  } >> "$SUMMARY_FILE"

  rm -f "$CLEAN_TMP"
done

echo
echo "=================================================="
echo " Outputs written to:"
echo "   ${SUMMARY_FILE}"
echo "   ${FULL_DIR}/<molecule>/stdout.txt"
echo "=================================================="
echo " Executable used:"
echo "   ${EXE}"
echo "=================================================="
