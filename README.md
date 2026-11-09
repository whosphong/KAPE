# KAPE: Kernel Adaptive Plasma Estimator v0.0.1 (WIP)
### Authors: Phong T. Ho, Abby Azari, Jay Newby, Nidhi Hegde

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

> [!NOTE]
> **Work in Progress (WIP):** **KAPE** is currently under active early development (`v0.0.1`). APIs, internal modules, and mathematical pipelines are undergoing frequent updates. Features and documentation will expand as validation against spacecraft datasets progresses.


**KAPE** (*Kernel Adaptive Plasma Estimator*) is a nonparametric regression framework built on Nadaraya-Watson kernel regression, engineered specifically for the non-uniform sampling geometries of top-hat electrostatic analyzers (ESAs) aboard heliospheric spacecraft.

By modeling sparse, discrete in-situ particle count observations, KAPE produces a continuous 3D reconstruction of particle velocity distribution functions (VDFs, $`f(\mathbf{v})`$) alongside key fluid and kinetic macroscopic moments.

---

## Key Features

- **Continuous 3D VDF Reconstruction:** Interpolates and reconstructs continuous phase space density $f(\mathbf{v})$ from discretized, sparse instrument counts without assuming an a priori distribution shape (e.g., Maxwellian or Bi-Maxwellian).
- **Dual Coordinate Frameworks:** Native support for reconstruction, transformation, and evaluation across:
  * **IAC:** Instrument-Aligned Coordinates
  * **MFAC:** Magnetic Field-Aligned Coordinates (rotated relative to local vector $\mathbf{B}$)
- **Automated Macroscopic Moments:** Calculates integrated fluid and kinetic parameters directly from reconstructed continuous distributions.
- **Multi-Mission Architecture:** Built to accommodate specialized energy-angle bin matrices and field-of-view masks across modern space plasma instrumentation.

---

## Instrument & Mission Support

| Mission | Instrument | Target Species | Coordinate Systems | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Parker Solar Probe (PSP)** | SPAN-I (Solar Probe ANalyzer - Ions) | Proton / Ion VDFs ($i^+$) | IAC, MFAC | **Supported (Active)** |
| **Parker Solar Probe (PSP)** | SPAN-E (Solar Probe ANalyzer - Electrons) | Electron VDFs ($e^-$) | IAC, MFAC | *In Development* |
| **Solar Orbiter (SolO)** | SWA-PAS (Proton Alpha Sensor) | Proton / Alpha VDFs ($i^+$) | IAC, MFAC | *In Development* |

---

## Macroscopic Moments Calculated

KAPE integrates the reconstructed 3D distribution $f(\mathbf{v})$ to derive standard macroscopic parameters:

| Parameter | Symbol | Units | Description |
| :--- | :--- | :--- | :--- |
| **Number Density** | $n$ | $\text{cm}^{-3}$ | Zeroth moment ($\int f(\mathbf{v}) \, d^3v$) |
| **Bulk Velocity Vector** | $\mathbf{u} = [v_i, v_j, v_k]$ | $\text{km/s}$ | First moment vector |
| **Bulk Speed** | $\vert{}\vert{}\mathbf{u}\vert{}\vert{}$ | $\text{km/s}$ | Vector magnitude of bulk flow |
| **Parallel Temperature** | $T_\parallel$ | $\text{eV}$ / $\text{K}$ | Thermal energy parallel to local magnetic field $\mathbf{B}$ |
| **Perpendicular Temperature**| $T_\perp$ | $\text{eV}$ / $\text{K}$ | Thermal energy perpendicular to local magnetic field $\mathbf{B}$ |

---

## Limitations

- **Field-of-View (FOV) Boundaries:** Unsampled regions of velocity space located far beyond physical instrument FOV limits require penalty regularizers to prevent unphysical extrapolation.
- **Computational Scaling:** Pairwise distance matrices for physical bandwidth step calculations scale quadratically ($\mathcal{O}(N^2)$) with particle bin count.
- **Sparse Count Thresholding:** Reconstructions in low-count or high-energy tail regions depend on baseline noise suppression ($Q_0$) to maintain physical stability.

---

## Development Roadmap & Status (v1.0.0 Release)

> **Status Legend:**  
> 🟩 **Completed** &nbsp;|&nbsp; 🟪 **Polishing** &nbsp;|&nbsp; 🟦 **In Development** &nbsp;|&nbsp; 🟧 **Paused** &nbsp;|&nbsp; 🟥 **Not Started**

---

### Core Estimator & Mathematics

> 🟩 **Nadaraya-Watson Estimator:** Core formulation for non-parametric kernel regression on 3D velocity distribution data from ESA.  
> 🟩 **Spherical Spatial Adaptive Bandwidth ($h$):** Physical nearest-neighbor step calculation ($r, \theta, \phi$) with float underflow guards.  
> 🟦 **Boundary Extrapolation Regularizer:** Penalized kernel routine to handle unsampled region extrapolation far from instrument boundaries using prior knowledge of zero density in higher velocity regions.  
> 🟩 **Automated Cross-Validation:** $k$-fold and leave-one-out optimization routines for hyperparameter tuning for bandwidth scaling.

---

### Macroscopic Moments & Physics

> 🟩 **Zeroth Moment Integration:** Particle number density ($n$).  
> 🟩 **First Moment Integration:** Bulk velocity vector ($\mathbf{u}$) and scalar flow speed.  
> 🟩 **Second Moment Integration:** Parallel ($T_\parallel$) and perpendicular ($T_\perp$) temperatures.  
> 🟩 **Coordinate Transformations:** Dynamic rotation between IAC and MFAC frames via local magnetic field vector $\mathbf{B}$.

---

### Visualization & Diagnostics

> 🟩 **Discrete Data Scatter in ($v_\parallel, v_\perp$):** Discrete data samples from ESA projected onto 2D plot via gyrotropic assumption.  
> 🟩 **Reconstructed Plasma VDF Contours in ($v_\parallel, v_\perp$):** Iso-density contour plot of reconstructed 3D VDF projected onto 2D plot via gyrotropic assumption.  
> 🟩 **Continuous Gyrotropic Surface in ($v_\parallel, v_\perp$):** Continuous color-density plot of reconstructed 3D VDF projected onto 2D plot via gyrotropic assumption.  
> 🟪 **Plot Layout & Export:** Multi-format figure export (.png, .svg, .pdf) with publication-ready labeling and visualization styling.

---

### API / User Experience / Quality of Life

> 🟦 **Automated Internal Bandwidth Engine:** Integrated spherical step bandwidth calculation ($h$) directly inside estimator logic, removing user pre-computation boilerplate.  
> 🟦 **Scikit-Learn Style High-Level API:** Intuitive `KAPE.fit()` and `KAPE.predict()` wrapper interface for general space plasma physics users.  
> 🟦 **PyPI Packaging & Distribution:** `pyproject.toml` configuration and build setup for standard `pip install kape` support.  
> 🟥 **Interactive Tutorials & Notebooks:** Google Colab and Jupyter example notebooks covering PSP SPAN-I CDF ingestion, VDF reconstruction, and moment evaluation.  
> 🟥 **Data Ingestion Helpers:** Direct loader utilities for NASA CDF files (PSP SPAN-I L2/L3).  
> 🟥 **Repository Documentation & API Reference:** Online documentation, Sphinx/MkDocs build, usage guides, and repository contribution standards.

---

## Installation

Install the latest development version directly from GitHub:

```bash
pip install git+[https://github.com/phongth/kape.git](https://github.com/whosphong/kape.git)
