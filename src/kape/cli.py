from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

"""
kape \
  --l2 /home/phongth/projects/def-aazari/phongth/data/PSPDat/sweap/spi/L2/spi_sf00/2022/09/psp_swp_spi_sf00_L2_8Dx32Ex8A_20220904_v04.cdf \
  --l3 /home/phongth/projects/def-aazari/phongth/data/PSPDat/sweap/spi/L3/spi_sf00/2022/09/psp_swp_spi_sf00_L3_mom_20220904_v04.cdf \
  --output_dir /home/phongth/projects/def-aazari/phongth/KAPE_out \
  --idx_min 11325 \
  --idx_max 11325 \
  --idx_step 10 \
  --domain_mode buffer \
  --buffer_angle_deg 10.0 \
  --buffer_r 5.0 \
  --plot
"""

def main() -> None:
  t0_total = time.perf_counter()
  default_progress = sys.stdout.isatty()

  parser = argparse.ArgumentParser(description="KAPE: PSP SPAN-I VDF Reconstruction and Moment Pipeline")
  parser.add_argument("--l2", type=Path, required=True)
  parser.add_argument("--l3", type=Path, required=True)
  parser.add_argument("--output_dir", type=Path, required=True)
  parser.add_argument("--idx_min", type=int, required=True)
  parser.add_argument("--idx_max", type=int, required=True)
  parser.add_argument("--idx_step", type=int, default=1)

  parser.add_argument("--R_sun", type=float, default=695700000.0)
  parser.add_argument("--R_max", type=float, default=40.0)

  parser.add_argument("--domain_mode", type=str, choices=["fov", "buffer", "full"], default="buffer")
  parser.add_argument("--buffer_r", type=float, default=5.0)
  parser.add_argument("--buffer_angle_deg", type=float, default=10.0)

  parser.add_argument("--regularizer", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--reg_pct", type=float, default=95.0)

  parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--save_svg", action=argparse.BooleanOptionalAction, default=False)

  parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=default_progress, help="Enable or disable tqdm progress bars (Default: True on TTY, False when redirected to Slurm log files).")

  args = parser.parse_args()

  import cdflib
  import numpy as np
  import pandas as pd
  from tqdm import tqdm

  from .bandwidth import compute_bandwidth_from_array, opt_alph
  from .core import KAPE, get_domain_bounds
  from .instruments.span_i import calcRMatrix, createVDF
  from .moments import EV_TO_K, compute_moments
  from .plotting import gyro_cont, gyro_contour, gyro_fhat_grid_spherical_domain, plot_gyro_scatter_original

  args.output_dir.mkdir(parents=True, exist_ok=True)

  L2Data = cdflib.CDF(str(args.l2))
  L3Data = cdflib.CDF(str(args.l3))

  info = L3Data.cdf_info()
  raw_vars = info.zVariables if hasattr(info, "zVariables") else info["zVariables"]
  l3_vars_map = {v.upper(): v for v in raw_vars}

  qFlags_day, vdf_day, vdf_one_count_day, counts_day, vx_day, vy_day, vz_day = createVDF(L2Data)
  A_day = calcRMatrix(L3Data)
  eligibleFilter = (L3Data["SUN_DIST"] / (args.R_sun / 10**3)) < args.R_max
  indexes = np.arange(args.idx_min, args.idx_max + 1, args.idx_step)

  idx, ts = [], []
  n_l3, n_kape = [], []
  u0_l3, u1_l3, ua_l3 = [], [], []
  u0_kape, u1_kape, ua_kape = [], [], []
  umag_l3, umag_kape = [], []
  T_par_l3_list, T_perp_l3_list = [], []
  T_par_kape_list, T_perp_kape_list = [], []

  buffer_angle_rad = np.radians(args.buffer_angle_deg)

  index_iter = tqdm(indexes, desc="VDF Sweeps", unit="vdf", disable=not args.progress)

  for i in index_iter:
    t0_idx = time.perf_counter()

    qFlags, vdf, vdf_one_count, vx, vy, vz, counts = (
        qFlags_day[eligibleFilter][i],
        vdf_day[eligibleFilter][i],
        vdf_one_count_day[eligibleFilter][i],
        vx_day[eligibleFilter][i],
        vy_day[eligibleFilter][i],
        vz_day[eligibleFilter][i],
        counts_day[eligibleFilter][i],
    )

    counts[:, [0], :] = 0
    vdf[:, [0], :] = 0
    vdf[counts <= 1] = 0

    vdfDate = pd.to_datetime(L2Data["TIME"][eligibleFilter][i], unit="s")
    ts_str = vdfDate.strftime("%Y-%m-%dT%H:%M:%S")

    QT = A_day[i]
    Q = QT.T
    vx, vy, vz = vx.ravel(), vy.ravel(), vz.ravel()
    X = np.column_stack([vx, vy, vz])
    fv = vdf.ravel()

    # 1. Fit Kernel Adaptive Plasma Estimator
    t0_fit = time.perf_counter()
    h_d = compute_bandwidth_from_array(X)
    alph, _, _ = opt_alph(X, fv, h_d, n_splits=25, show_progress=args.progress)
    print(f"Best alpha: {alph:.4f}")
    model = KAPE(X, fv, h_d * alph).fit()
    t_fit = time.perf_counter() - t0_fit

    # 2. Compute Macroscopic Fluid & Kinetic Moments
    t0_mom = time.perf_counter()
    nhat, u_car, u_fac, T_par_kape, T_perp_kape = compute_moments(
        model, X, Q, buffer_r=args.buffer_r, buffer_angle_rad=buffer_angle_rad, n_r=60, n_theta=60, n_phi=60, chunk_size=50000
    )
    t_mom = time.perf_counter() - t0_mom

    u0, u1, ua = np.float64(u_fac[0]), np.float64(u_fac[1]), np.float64(u_fac[2])

    dens = float(L3Data["DENS"][i])
    vel_inst = np.array(L3Data["VEL_INST"][i])
    A_all = calcRMatrix({"MAGF_INST": np.array(L3Data["MAGF_INST"])})
    v_fac = A_all[i] @ vel_inst

    t_par_l3_val, t_perp_l3_val = np.nan, np.nan
    tensor_key = None
    for target in ["T_TENSOR_INST", "T_TENSOR", "TEMP_TENSOR", "T_TENSOR_SC", "T_TENSOR_RTN"]:
      if target in l3_vars_map:
        tensor_key = l3_vars_map[target]
        break

    if tensor_key is not None:
      T_raw = np.array(L3Data[tensor_key][i])
      B_vec = np.array(L3Data["MAGF_INST"][i])
      B_norm = np.linalg.norm(B_vec)

      if B_norm > 0:
        b_hat = B_vec / B_norm
        if T_raw.size == 6:
          txx, tyy, tzz, txy, txz, tyz = T_raw.ravel()
          T_mat = np.array([[txx, txy, txz], [txy, tyy, tyz], [txz, tyz, tzz]])
        elif T_raw.shape == (3, 3):
          T_mat = T_raw
        elif T_raw.size == 9:
          T_mat = T_raw.reshape(3, 3)
        else:
          T_mat = None

        if T_mat is not None:
          t_par_eV = float(b_hat @ T_mat @ b_hat)
          t_trace_eV = float(np.trace(T_mat))
          t_perp_eV = float(0.5 * (t_trace_eV - t_par_eV))
          t_par_l3_val = t_par_eV * EV_TO_K
          t_perp_l3_val = t_perp_eV * EV_TO_K

    if np.isnan(t_par_l3_val) or np.isnan(t_perp_l3_val):
      if "TEMP" in l3_vars_map:
        t_val_eV = float(L3Data[l3_vars_map["TEMP"]][i])
        t_par_l3_val = t_val_eV * EV_TO_K
        t_perp_l3_val = t_val_eV * EV_TO_K

    v_fac_ordered = np.array([v_fac[0], v_fac[2], v_fac[1]])
    u_mag_l3_val = np.linalg.norm(v_fac_ordered)
    u_mag_kape_val = np.linalg.norm(u_fac)

    aniso_l3 = t_perp_l3_val / t_par_l3_val if t_par_l3_val and t_par_l3_val > 0 else np.nan
    aniso_kape = T_perp_kape / T_par_kape if T_par_kape > 0 else np.nan

    diff_n = ((nhat - dens) / dens) * 100.0 if dens != 0 else np.nan
    diff_umag = ((u_mag_kape_val - u_mag_l3_val) / u_mag_l3_val) * 100.0 if u_mag_l3_val != 0 else np.nan
    diff_tpar = ((T_par_kape - t_par_l3_val) / t_par_l3_val) * 100.0 if t_par_l3_val != 0 else np.nan
    diff_tperp = ((T_perp_kape - t_perp_l3_val) / t_perp_l3_val) * 100.0 if t_perp_l3_val != 0 else np.nan
    diff_u0 = ((u_fac[0] - v_fac_ordered[0]) / abs(v_fac_ordered[0])) * 100.0 if v_fac_ordered[0] != 0 else np.nan
    diff_u1 = ((u_fac[1] - v_fac_ordered[1]) / abs(v_fac_ordered[1])) * 100.0 if v_fac_ordered[1] != 0 else np.nan
    diff_ua = ((u_fac[2] - v_fac_ordered[2]) / abs(v_fac_ordered[2])) * 100.0 if v_fac_ordered[2] != 0 else np.nan
    diff_aniso = ((aniso_kape - aniso_l3) / aniso_l3) * 100.0 if not np.isnan(aniso_l3) and aniso_l3 != 0 else np.nan

    n_l3.append(dens)
    n_kape.append(nhat)
    u0_l3.append(v_fac_ordered[0])
    u1_l3.append(v_fac_ordered[1])
    ua_l3.append(v_fac_ordered[2])
    umag_l3.append(u_mag_l3_val)
    u0_kape.append(u0)
    u1_kape.append(u1)
    ua_kape.append(ua)
    umag_kape.append(u_mag_kape_val)
    T_par_l3_list.append(t_par_l3_val)
    T_perp_l3_list.append(t_perp_l3_val)
    T_par_kape_list.append(T_par_kape)
    T_perp_kape_list.append(T_perp_kape)

    idx.append(i)
    ts.append(ts_str)

    # 3. Plotting
    t_plot = 0.0
    if args.plot:
      t0_plot = time.perf_counter()
      vrot = Q.T @ np.array([vx, vy, vz])
      V_mag = np.column_stack([vrot[0], vrot[2], vrot[1]])

      r_min, r_max, theta_min, theta_max, phi_min, phi_max = get_domain_bounds(
          X, domain_mode=args.domain_mode, buffer_r=args.buffer_r, buffer_angle_rad=buffer_angle_rad
      )
      buffer_bounds = get_domain_bounds(
          X, domain_mode="buffer", buffer_r=args.buffer_r, buffer_angle_rad=buffer_angle_rad
      )

      if args.regularizer: model.support_Q0(pct=args.reg_pct)
      else: model.Q0 = None

      vperp_grid = np.linspace(0.0, r_max * 1.2, 151)
      vpar_grid = np.linspace(-r_max * 1.2, r_max * 1.2, 221)

      Fgyro, domain_mask = gyro_fhat_grid_spherical_domain(
          model=model,
          u_fac=u_fac,
          Q=Q,
          r_min=r_min,
          r_max=r_max,
          theta_min=theta_min,
          theta_max=theta_max,
          phi_min=phi_min,
          phi_max=phi_max,
          domain_mode=args.domain_mode,
          n_theta=16,
          vperp_grid=vperp_grid,
          vpar_grid=vpar_grid,
          show_progress=args.progress,
          buffer_bounds=buffer_bounds,
      )

      base_path = args.output_dir / f"Plot_3D_scatter_{ts_str}.png"
      plot_gyro_scatter_original(V=V_mag, f=fv, u=u_fac, save_path=base_path)
      if args.save_svg: plot_gyro_scatter_original(V=V_mag, f=fv, u=u_fac, save_path=base_path.with_suffix(".svg"))

      base_contour = args.output_dir / f"Plot_Gyro_Contour_{ts_str}.png"
      gyro_contour(Fhat=Fgyro, domain_mask=domain_mask, vperp_grid=vperp_grid, vpar_grid=vpar_grid, save_path=base_contour)
      if args.save_svg: gyro_contour(Fhat=Fgyro, domain_mask=domain_mask, vperp_grid=vperp_grid, vpar_grid=vpar_grid, save_path=base_contour.with_suffix(".svg"))

      base_cont = args.output_dir / f"Plot_Gyro_Continuous_{ts_str}.png"
      gyro_cont(Fhat=Fgyro, domain_mask=domain_mask, vperp_grid=vperp_grid, vpar_grid=vpar_grid, save_path=base_cont)
      if args.save_svg: gyro_cont(Fhat=Fgyro, domain_mask=domain_mask, vperp_grid=vperp_grid, vpar_grid=vpar_grid, save_path=base_cont.with_suffix(".svg"))

      t_plot = time.perf_counter() - t0_plot

    t_idx_total = time.perf_counter() - t0_idx

    # Console Terminal Summary Table
    print("=" * 86)
    print(f" INDEX {i:<6} | Timestamp: {ts_str}")
    print("=" * 86)
    print(f" {'Moments':<22} | {'SPANi L3':<18} | {'KAPE':<18} | {'Diff (%)':<10}")
    print("-" * 86)
    print(f" {'n [cm⁻³]':<22} | {dens:<18.4f} | {nhat:<18.4f} | {diff_n:+9.2f}%")
    print(f" {'u_⊥1 [km/s]':<22} | {v_fac_ordered[0]:<18.2f} | {u_fac[0]:<18.2f} | {diff_u0:+9.2f}%")
    print(f" {'u_⊥2 [km/s]':<22} | {v_fac_ordered[1]:<18.2f} | {u_fac[1]:<18.2f} | {diff_u1:+9.2f}%")
    print(f" {'u_∥  [km/s]':<22} | {v_fac_ordered[2]:<18.2f} | {u_fac[2]:<18.2f} | {diff_ua:+9.2f}%")
    print(f" {'Speed |u|  [km/s]':<22} | {u_mag_l3_val:<18.2f} | {u_mag_kape_val:<18.2f} | {diff_umag:+9.2f}%")
    print(f" {'T_∥  [K]':<22} | {t_par_l3_val:<18.2f} | {T_par_kape:<18.2f} | {diff_tpar:+9.2f}%")
    print(f" {'T_⊥  [K]':<22} | {t_perp_l3_val:<18.2f} | {T_perp_kape:<18.2f} | {diff_tperp:+9.2f}%")
    print(f" {'(T_⊥/T_∥)':<22} | {aniso_l3:<18.3f} | {aniso_kape:<18.3f} | {diff_aniso:+9.2f}%")
    print("-" * 86)
    print(f" [TIMING] Fit: {t_fit:.2f}s | Integration: {t_mom:.2f}s | Plot: {t_plot:.2f}s | Total: {t_idx_total:.2f}s")
    print("=" * 86)

  df_out = pd.DataFrame({
      "index": idx,
      "timestamp": ts,
      "n_l3_cm3": n_l3,
      "n_kape_cm3": n_kape,
      "diff_n_pct": [((n - l) / l) * 100 for n, l in zip(n_kape, n_l3)],
      "u_perp1_l3_kms": u0_l3,
      "u_perp2_l3_kms": u1_l3,
      "u_par_l3_kms": ua_l3,
      "umag_l3_kms": umag_l3,
      "u_perp1_kape_kms": u0_kape,
      "u_perp2_kape_kms": u1_kape,
      "u_par_kape_kms": ua_kape,
      "umag_kape_kms": umag_kape,
      "diff_umag_pct": [((n - l) / l) * 100 for n, l in zip(umag_kape, umag_l3)],
      "T_par_l3_K": T_par_l3_list,
      "T_perp_l3_K": T_perp_l3_list,
      "anisotropy_l3": [tp / tpa if tpa > 0 else np.nan for tp, tpa in zip(T_perp_l3_list, T_par_l3_list)],
      "T_par_kape_K": T_par_kape_list,
      "T_perp_kape_K": T_perp_kape_list,
      "anisotropy_kape": [tp / tpa if tpa > 0 else np.nan for tp, tpa in zip(T_perp_kape_list, T_par_kape_list)],
      "diff_T_par_pct": [((n - l) / l) * 100 for n, l in zip(T_par_kape_list, T_par_l3_list)],
      "diff_T_perp_pct": [((n - l) / l) * 100 for n, l in zip(T_perp_kape_list, T_perp_l3_list)],
  })

  csv_output_path = args.output_dir / f"moments_{args.idx_min}_to_{args.idx_max}.csv"
  df_out.to_csv(csv_output_path, index=False)
  print(f"\n[Success] Moments comparison saved to: {csv_output_path}\n[Total Program Execution Time]: {time.perf_counter() - t0_total:.2f}s")


if __name__ == "__main__":
  main()