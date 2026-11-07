from __future__ import annotations
import numpy as np
from scipy.integrate import simpson
from .core import get_domain_bounds, in_spherical_domain

# Physical Constants
M_PROTON = 1.67262192e-27  # Proton mass [kg]
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]
KM2_TO_M2 = 1e6  # (1 km/s)^2 = 1e6 (m/s)^2
T_FACTOR = (M_PROTON * KM2_TO_M2) / K_BOLTZMANN  # ~121.15 K / (km/s)^2
EV_TO_K = 11604.525  # eV to Kelvin


def compute_moments(
    model,
    X: np.ndarray,
    Q: np.ndarray, # B ROTATION MATRIX
    buffer_r: float = 5.0, # OPTIMAL DEFAULT FOUND
    buffer_angle_rad: float = np.radians(10.0), # OPTIMAL DEFAULT FOUND
    n_r: int = 60,
    n_theta: int = 60,
    n_phi: int = 60,
    chunk_size: int = 50000) -> tuple[float, np.ndarray, np.ndarray, float, float]:

    """
    moment calculation function for KAPE. computes the density, bulk velocity, and parallel/perpendicular temperatures from the KAPE model.
    Compute initially in spherical coordinates then transformed into cartesian coordinates using change of varible formula. outputs can be in either cartesian or field-aligned coordinates depending on the Q matrix provided. Q is the rotation matrix that transforms from cartesian to field-aligned coordinates.
    """

  r_min, r_max, theta_min, theta_max, phi_min, phi_max = get_domain_bounds(X, domain_mode="buffer", buffer_r=buffer_r, buffer_angle_rad=buffer_angle_rad)

  r_grid = np.linspace(r_min, r_max, n_r)
  theta_grid = np.linspace(theta_min, theta_max, n_theta)
  phi_grid = np.linspace(phi_min, phi_max, n_phi)

  cos_th, sin_th = np.cos(theta_grid), np.sin(theta_grid)
  cos_ph, sin_ph = np.cos(phi_grid), np.sin(phi_grid)

  r_3d = r_grid[:, None, None] # 3D grid of r values for integration. shape (n_r, 1, 1)
  cos_th_3d, sin_th_3d = cos_th[None, :, None], sin_th[None, :, None] # 3D grid of theta values for integration. shape (1, n_theta, 1)
  cos_ph_3d, sin_ph_3d = cos_ph[None, None, :], sin_ph[None, None, :] # 3D grid of phi values for integration. shape (1, 1, n_phi)

  vx_grid = r_3d * cos_ph_3d * cos_th_3d # 3D grid of vx values for integration. shape (n_r, n_theta, n_phi)
  vy_grid = r_3d * cos_ph_3d * sin_th_3d # 3D grid of vy values for integration. shape (n_r, n_theta, n_phi)
  vz_grid = np.broadcast_to(r_3d * sin_ph_3d, (n_r, n_theta, n_phi)) # 3D grid of vz values for integration. shape (n_r, n_theta, n_phi)

  X_grid = (np.stack([vx_grid, vy_grid, vz_grid], axis=-1).reshape(-1, 3).astype(np.float32)) # MADE AN INTEGRATION GRID.

  inside = in_spherical_domain(X_grid,r_min,r_max,theta_min,theta_max,phi_min,phi_max,domain_mode="buffer") # filter down integration grids

  fhat_flat = np.zeros(X_grid.shape[0], dtype=np.float32)
  inside_idx = np.flatnonzero(inside)

  if len(inside_idx) > 0:
    for start in range(0, len(inside_idx), chunk_size):
      end = start + chunk_size
      batch = inside_idx[start:end]
      fhat_flat[batch] = model.predict(X_grid[batch], prior=False)

  fhat_grid = fhat_flat.reshape(n_r, n_theta, n_phi)
  jacobian = (r_grid**2)[:, None, None] * cos_ph_3d
  fj = fhat_grid * jacobian

  # 1st Pass: Density & Bulk Velocity
  integrands_1st = np.stack([fj, fj * vx_grid, fj * vy_grid, fj * vz_grid], axis=0)
  i_phi = simpson(integrands_1st, x=phi_grid, axis=3)
  i_th = simpson(i_phi, x=theta_grid, axis=2)
  totals = simpson(i_th, x=r_grid, axis=1)

  n_raw, I_vx, I_vy, I_vz = totals
  n_cm3 = n_raw * 1e-6

  inv_n_raw = 1.0 / n_raw if n_raw > 0 else 1.0
  u_car = np.array([I_vx, I_vy, I_vz]) * inv_n_raw
  u_fac = (Q.T @ u_car)[[0, 2, 1]]

  # 2nd Pass: Parallel & Perpendicular Temperatures
  vprime_x = vx_grid - u_car[0]
  vprime_y = vy_grid - u_car[1]
  vprime_z = vz_grid - u_car[2]

  b_hat = Q[:, 1]
  vprime_par = vprime_x * b_hat[0] + vprime_y * b_hat[1] + vprime_z * b_hat[2]
  vprime_sq_total = vprime_x**2 + vprime_y**2 + vprime_z**2
  vprime_sq_perp = np.maximum(vprime_sq_total - vprime_par**2, 0.0)

  integrands_2nd = np.stack(
      [fj * (vprime_par**2), fj * vprime_sq_perp], axis=0
  )
  i_phi_2 = simpson(integrands_2nd, x=phi_grid, axis=3)
  i_th_2 = simpson(i_phi_2, x=theta_grid, axis=2)
  I_vpar2, I_vperp2 = simpson(i_th_2, x=r_grid, axis=1)

  T_par = T_FACTOR * (I_vpar2 * inv_n_raw)
  T_perp = 0.5 * T_FACTOR * (I_vperp2 * inv_n_raw)


    # u_car = [vx, vy, vz] in cartesian coordinates
    # u_fac = [vperp, vpar, vperp] in field aligned coordinates
    # T_par = parallel temperature in Kelvin
    # T_perp = perpendicular temperature in Kelvin
  return n_cm3, u_car, u_fac, T_par, T_perp