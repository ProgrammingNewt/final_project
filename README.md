# Molecule Band Gaps and Dipoles Final Project

For ease of use we have streamlined building the project, running all molecular input files, and collecting both **clean summaries** and **full debug output** for each molecule.

---

## Overview of structure

The script:

1. Cleans the build environment  
2. Builds the project  
3. Runs the executable once per molecule (`*.json` input)  
4. Collects:
   - A **summary** of MO energies, HOMO–LUMO gaps, and dipoles
   - The **full raw stdout** for each molecule, saved separately in case you need to see the full outputs.   

All outputs are placed under a single directory for easy inspection and cleanup (./clean.sh will remove this file too). 

---

## How to Run

From the project root:

```bash
chmod +x run_all.sh
./run_all.sh
