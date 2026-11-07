# KAPE: Kernel Adaptive Plasma Estimator v0.0.1 (WIP)

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

### 🧮 Core Estimator & Mathematics

| Feature | Description | Status |
| :--- | :--- | :---: |
| **Nadaraya-Watson Estimator** | Core non-parametric kernel regression engine for 3D ESA VDF data | 🟩 |
| **Spherical Spatial Bandwidth** | Physical nearest-neighbor step calculation ($r, \theta, \phi$) with underflow guards | 🟩 |
| **Automated Cross-Validation** | $k$-fold and LOO optimization routines for hyperparameter tuning | 🟩 |
| **Boundary Extrapolation Regularizer** | Penalized kernel routine using high-velocity zero-density priors | 🟦 |

---

### 🌌 Macroscopic Moments & Physics

| Feature | Description | Status |
| :--- | :--- | :---: |
| **Zeroth Moment Integration** | Number density ($n$) integration ($\int f(\mathbf{v}) \, d^3v$) | 🟩 |
| **First Moment Integration** | Bulk velocity vector ($\mathbf{u}$) and scalar flow speed | 🟩 |
| **Second Moment Integration** | Parallel ($T_\parallel$) and perpendicular ($T_\perp$) thermal energy derivation | 🟩 |
| **Coordinate Transformations** | Dynamic frame rotation between IAC and MFAC via local magnetic field $\mathbf{B}$ | 🟩 |

---

### 📊 Visualization & Diagnostics

| Feature | Description | Status |
| :--- | :--- | :---: |
| **2D Discrete Scatter Mapping** | Discrete ESA count samples projected onto ($v_\parallel, v_\perp$) plane under gyrotropy | 🟩 |
| **Reconstructed 2D VDF Contours** | Iso-density contour slice rendering of reconstructed 3D VDFs in ($v_\parallel, v_\perp$) | 🟩 |
| **Continuous Gyrotropic Maps** | Continuous spatial density surface projections in ($v_\parallel, v_\perp$) space | 🟩 |
| **Publication Plotting Layout** | Multi-format figure exporter (.png, .svg, .pdf) with standard styling | 🟪 |

---

### 🛠️ API, User Experience & Documentation

| Feature | Description | Status |
| :--- | :--- | :---: |
| **Automated Internal Bandwidth Engine** | Internal calculation of $h$ inside `.fit()`, removing user boilerplate | 🟦 |
| **Scikit-Learn Style High-Level API** | Intuitive `KAPE.fit()` and `KAPE.predict()` wrapper interface | 🟦 |
| **PyPI Packaging & Distribution** | Standard `pyproject.toml` build setup for `pip install kape` | 🟦 |
| **Interactive Tutorials & Notebooks** | Google Colab and Jupyter notebooks covering PSP SPAN-I workflow | 🟥 |
| **Data Ingestion Helpers** | Direct loader utilities for NASA PSP SPAN-I (L2/L3) CDF files | 🟥 |
| **Repository & API Reference** | Sphinx/MkDocs web documentation, usage guides, and contribution standards | 🟥 |

---

## Installation

Install the latest development version directly from GitHub:

```bash
pip install git+[https://github.com/phongth/kape.git](https://github.com/whosphong/kape.git)
