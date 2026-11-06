# KAPE: Kernel Adaptive Plasma Estimator v0.0.1 (WIP)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

> [!NOTE]
> **Work in Progress (WIP):** **KAPE** is currently under active early development (`v0.0.1`). APIs, internal modules, and mathematical pipelines are undergoing frequent updates. Features and documentation will expand as validation against spacecraft datasets progresses.


**KAPE** (*Kernel Adaptive Plasma Estimator*) is a nonparametric regression framework built on Nadaraya-Watson kernel regression, engineered specifically for the non-uniform sampling geometries of top-hat electrostatic analyzers (ESAs) aboard heliospheric spacecraft.

By modeling sparse, discrete in-situ particle count observations, KAPE produces a continuous 3D reconstruction of particle velocity distribution functions (VDFs, $f(\mathbf{v})$) alongside key fluid and kinetic macroscopic moments.
---

## Key Features

* **Continuous 3D VDF Reconstruction:** Interpolates and reconstructs continuous phase space density $f(\mathbf{v})$ from discretized, sparse instrument counts without assuming an a priori distribution shape (e.g., Maxwellian or Bi-Maxwellian).
* **Dual Coordinate Frameworks:** Native support for reconstruction, transformation, and evaluation across:
  * **IAC:** Instrument-Aligned Coordinates
  * **MFAC:** Magnetic Field-Aligned Coordinates (rotated relative to local vector $\mathbf{B}$)
* **Automated Macroscopic Moments:** Calculates integrated fluid and kinetic parameters directly from reconstructed continuous distributions.
* **Multi-Mission Architecture:** Built to accommodate specialized energy-angle bin matrices and field-of-view masks across modern space plasma instrumentation.

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

In progress...

---

## Installation

In progress...

---
