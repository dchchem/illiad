# ILLIAD - Precise Layered Molecular Packing Metrics Evaluation
<img width="500" alt="illiad_cover_github" src="https://github.com/user-attachments/assets/b06b960c-02a3-460e-a36f-7cb9acb6566e" />

ILLIAD is a Python script dedicated to a precise calculation of various descriptors and metrics for crystal structures with a layered packing. The script can be run from command line both with and without initial parsing.
The program reads the SHELXL ```.res``` file with the asymmetric unit cell **already grown** (if the molecule lies on the special position, e.g. mirror plane, inversion center, rotoational axis, etc.) and prints out some crucial intra- and interlayered packing metrics.

Example of command line execution:

- ```-h``` - print the usage instructions
- ```-hkl``` - Miller (hkl) indices for a layer (need to be calculated by user a priori, e.g. with Mercury, XP or Olex2)
- ```-m``` - number of molecules on layer per unit cell
- ```-r``` - van der Waals sphere resolution (number of vertices per sphere)
- ```-q``` - quiet mode with no verbose until final results are printed
- ```-sect``` - plot the layer cross-section of a molecular van der Waals surface
- ```-surf``` - plot the molecular van der Waals surface
- ```-scld``` - plot the point cloud for molecular van der Waals surface
- ```-p``` - project name
```
python illiad.py [-h] [-hkl h k l] [-m MOL] [-r RESOLUTION] [-q] [--sect SECT] [--surf SURF] [--scld SCLD] [-p PROJECT]
```

<div align="right">
  (C) Danila R. Chernyavskiy, 2026
