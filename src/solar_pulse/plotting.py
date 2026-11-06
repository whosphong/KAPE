from __future__ import annotations
from pathlib import Path
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .core import in_spherical_domain

matplotlib.use("Agg")

VMIN, VMAX = -3.0, 3.0
LABEL_FS, TITLE_FS, TICK_FS, CB_FS = 20, 20, 20, 15
ticks = np.arange(-3, 4, 1)


def plot_gyro_scatter_original(
    V: np.ndarray,
    f: np.ndarray,
    u: np.ndarray,
    save_path: Path | str | None = None,
) -> None:
  f = np.asarray(f, dtype=float)
  vp0, vp1, vpa = V[:, 0], V[:, 1], V[:, 2]
  u0, u1, ua = u

  vperp = np.sqrt((vp0 - u0) ** 2 + (vp1 - u1) ** 2)
  vpar = vpa - ua

  zero_mask = f <= 0
  f_log = np.log10(np.maximum(f, 1e-30))
  f_log[zero_mask] = np.nan

  vperp_sym = np.concatenate([vperp, -vperp])
  vpar_sym = np.concatenate([vpar, vpar])
  zero_sym = np.concatenate([zero_mask, zero_mask])
  f_sym = np.concatenate([f_log, f_log])

  nonzero_sym = ~zero_sym
  vperp_plot = vperp_sym[nonzero_sym]
  vpar_plot = vpar_sym[nonzero_sym]
  f_plot = f_sym[nonzero_sym]

  idx = np.argsort(f_plot)
  vperp_plot = vperp_plot[idx]
  vpar_plot = vpar_plot[idx]
  f_plot = f_plot[idx]

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.scatter(
      vperp_sym[zero_sym],
      vpar_sym[zero_sym],
      s=20,
      facecolors="white",
      edgecolors="black",
      linewidths=0.6,
      zorder=1,
  )
  sc = ax.scatter(
      vperp_plot,
      vpar_plot,
      c=f_plot,
      s=30,
      cmap="plasma",
      vmin=VMIN,
      vmax=VMAX,
      zorder=2,
  )
  ax.set_xlabel(r"$v_\perp$ [km/s]", fontsize=LABEL_FS)
  ax.set_ylabel(r"$v_\parallel$ [km/s]", fontsize=LABEL_FS)
  ax.set_xlim(-400, 400)
  ax.set_ylim(-500, 280)
  ax.set_axisbelow(True)
  ax.grid(True, color="black", linewidth=0.4, alpha=0.3)
  ax.tick_params(axis="both", labelsize=TICK_FS)

  cb = fig.colorbar(sc, ax=ax, pad=0.02)
  cb.set_ticks(ticks)
  cb.set_ticklabels([rf"$10^{{{int(t)}}}$" for t in ticks], fontsize=TICK_FS)
  cb.set_label(r"$\hat{f}(v)\;[\mathrm{m^{-3}\,(km/s)^{-3}}]$", fontsize=CB_FS)

  fig.tight_layout()
  if save_path:
    p = Path(save_path)
    plt.savefig(p, dpi=300 if p.suffix == ".png" else None, bbox_inches="tight")
  plt.close(fig)


def gyro_fhat_grid_spherical_domain(
    model,
    u_fac: np.ndarray,
    Q: np.ndarray,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    domain_mode: str = "buffer",
    n_theta: int = 32,
    vperp_grid: np.ndarray | None = None,
    vpar_grid: np.ndarray | None = None,
    show_progress: bool = True,
    chunk_size: int = 500000,
    buffer_bounds: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
  ug0, ug1, uga = u_fac[0], u_fac[1], u_fac[2]

  if vperp_grid is None:
    vperp_grid = np.linspace(0.0, r_max, 151)
  if vpar_grid is None:
    vpar_grid = np.linspace(-r_max, r_max, 301)

  n_perp, n_par = len(vperp_grid), len(vpar_grid)

  thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
  ct, st = np.cos(thetas), np.sin(thetas)

  rp_3d = vperp_grid[:, None, None]
  vpar_3d = vpar_grid[None, :, None]
  ct_3d, st_3d = ct[None, None, :], st[None, None, :]

  vp0_all = np.broadcast_to(ug0 + rp_3d * ct_3d, (n_perp, n_par, n_theta))
  vp1_all = np.broadcast_to(ug1 + rp_3d * st_3d, (n_perp, n_par, n_theta))
  v2_all = np.broadcast_to(uga + vpar_3d, (n_perp, n_par, n_theta))

  Xq_fac = np.stack([vp0_all, v2_all, vp1_all], axis=-1).reshape(-1, 3)
  Xq_car = (Xq_fac @ Q.T).astype(np.float32, copy=False)

  inside = in_spherical_domain(
      Xq_car,
      r_min,
      r_max,
      theta_min,
      theta_max,
      phi_min,
      phi_max,
      domain_mode=domain_mode,
  )

  if buffer_bounds is not None and model.Q0 is not None:
    in_buffer = in_spherical_domain(
        Xq_car, *buffer_bounds, domain_mode="buffer"
    )
  else:
    in_buffer = np.ones(len(Xq_car), dtype=bool)

  y_pred = np.full(len(Xq_car), np.nan, dtype=np.float32)
  inside_indices = np.flatnonzero(inside)

  if len(inside_indices) > 0:
    batch_ranges = range(0, len(inside_indices), chunk_size)
    batch_iter = tqdm(
        batch_ranges,
        desc="  └─ Gyro Grid Evaluation",
        leave=False,
        disable=not show_progress,
    )
    for start in batch_iter:
      end = start + chunk_size
      batch_idx = inside_indices[start:end]
      b_in = in_buffer[batch_idx]
      if np.any(b_in):
        y_pred[batch_idx[b_in]] = model.predict(
            Xq_car[batch_idx[b_in]], use_prior=False
        )
      if np.any(~b_in):
        y_pred[batch_idx[~b_in]] = model.predict(
            Xq_car[batch_idx[~b_in]], use_prior=True
        )

  y_pred_3d = y_pred.reshape(n_perp, n_par, n_theta)
  inside_3d = inside.reshape(n_perp, n_par, n_theta)

  row_inside = inside_3d.any(axis=2)
  f_vals = np.nan_to_num(y_pred_3d, nan=0.0)

  F = f_vals.mean(axis=2)
  if domain_mode != "full":
    F[~row_inside] = 0.0

  domain_mask = ~row_inside if domain_mode != "full" else None
  return F, domain_mask


def gyro_contour(
    Fhat: np.ndarray,
    vperp_grid: np.ndarray,
    vpar_grid: np.ndarray,
    domain_mask: np.ndarray | None = None,
    nan_mask: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> None:
  Fhat_log = np.log10(np.abs(Fhat) + 1e-30)
  F_sym = np.vstack([np.flipud(Fhat_log), Fhat_log])
  vperp_sym = np.concatenate([-vperp_grid[::-1], vperp_grid])

  Xp, Yp = np.meshgrid(vperp_sym, vpar_grid, indexing="ij")
  levels = np.linspace(VMIN, VMAX, 17)

  cmap = plt.cm.plasma.copy()
  cmap.set_under(cmap(0.0))

  fig, ax = plt.subplots(figsize=(10, 8))

  cf = ax.contourf(Xp, Yp, F_sym, levels=levels, cmap=cmap, extend="min")

  # Soft, semi-transparent white isoline contours (alpha=0.45)
  ax.contour(
      Xp,
      Yp,
      F_sym,
      levels=levels,
      colors="white",
      linewidths=0.5,
      linestyles="solid",
      alpha=0.45,
  )

  mask_to_use = (
      domain_mask
      if domain_mask is not None
      else (nan_mask if nan_mask is not None else None)
  )
  if mask_to_use is not None and np.any(mask_to_use):
    mask_sym = np.vstack([np.flipud(mask_to_use), mask_to_use])
    # Distinct cyan dashed line for domain boundary
    ax.contour(
        Xp,
        Yp,
        mask_sym,
        levels=[0.5],
        colors="cyan",
        linewidths=1.2,
        linestyles="--",
        zorder=4,
    )

  ax.set_xlabel(r"$v_\perp$ [km/s]", fontsize=LABEL_FS)
  ax.set_ylabel(r"$v_\parallel$ [km/s]", fontsize=LABEL_FS)
  ax.set_xlim(-400, 400)
  ax.set_ylim(-500, 280)

  ax.set_axisbelow(False)
  ax.grid(True, color="white", linestyle="--", linewidth=0.5, alpha=0.3)
  ax.tick_params(axis="both", labelsize=TICK_FS)

  cb = fig.colorbar(cf, ax=ax, pad=0.02)
  cb.set_ticks(ticks)
  cb.set_ticklabels([rf"$10^{{{int(t)}}}$" for t in ticks], fontsize=TICK_FS)
  cb.set_label(r"$\hat{f}(v)\;[\mathrm{m^{-3}\,(km/s)^{-3}}]$", fontsize=CB_FS)

  fig.tight_layout()
  if save_path:
    p = Path(save_path)
    plt.savefig(p, dpi=300 if p.suffix == ".png" else None, bbox_inches="tight")
  plt.close(fig)


def gyro_cont(
    Fhat: np.ndarray,
    vperp_grid: np.ndarray,
    vpar_grid: np.ndarray,
    domain_mask: np.ndarray | None = None,
    nan_mask: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> None:
  Fhat_log = np.log10(np.abs(Fhat) + 1e-30)
  F_sym = np.vstack([np.flipud(Fhat_log), Fhat_log])
  vperp_sym = np.concatenate([-vperp_grid[::-1], vperp_grid])

  Xp, Yp = np.meshgrid(vperp_sym, vpar_grid, indexing="ij")
  levels = np.linspace(VMIN, VMAX, 17)

  mask_to_use = (
      domain_mask
      if domain_mask is not None
      else (nan_mask if nan_mask is not None else None)
  )
  nan_sym = (
      np.vstack([np.flipud(mask_to_use), mask_to_use])
      if mask_to_use is not None
      else np.zeros_like(F_sym, dtype=bool)
  )

  F_sym_m = np.ma.masked_where((F_sym < VMIN) | (F_sym > VMAX) | nan_sym, F_sym)

  cmap = plt.cm.plasma.copy()
  cmap.set_under(cmap(0.0))
  cmap.set_bad(cmap(0.0))

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.set_facecolor(cmap(0.0))

  norm = colors.Normalize(vmin=VMIN, vmax=VMAX)
  cf = ax.pcolormesh(Xp, Yp, F_sym_m, norm=norm, cmap=cmap, shading="auto")

  if mask_to_use is not None and np.any(mask_to_use):
    # Distinct cyan dashed line for domain boundary
    ax.contour(
        Xp,
        Yp,
        nan_sym,
        levels=[0.5],
        colors="cyan",
        linewidths=1.2,
        linestyles="--",
        zorder=5,
    )

  # Soft, semi-transparent white isoline overlay (alpha=0.45)
  ax.contour(
      Xp,
      Yp,
      F_sym,
      levels=levels,
      colors="white",
      linewidths=0.5,
      linestyles="solid",
      alpha=0.45,
      zorder=4,
  )

  ax.set_xlabel(r"$v_\perp$ [km/s]", fontsize=LABEL_FS)
  ax.set_ylabel(r"$v_\parallel$ [km/s]", fontsize=LABEL_FS)
  ax.set_xlim(-400, 400)
  ax.set_ylim(-500, 280)

  ax.set_axisbelow(False)
  ax.grid(True, color="white", linestyle="--", linewidth=0.5, alpha=0.3)
  ax.tick_params(axis="both", labelsize=TICK_FS)

  cb = fig.colorbar(cf, ax=ax, pad=0.02, extend="min")
  cb.set_ticks(ticks)
  cb.set_ticklabels([rf"$10^{{{int(t)}}}$" for t in ticks], fontsize=TICK_FS)
  cb.set_label(r"$\hat{f}(v)\;[\mathrm{m^{-3}\,(km/s)^{-3}}]$", fontsize=CB_FS)

  fig.tight_layout()
  if save_path:
    p = Path(save_path)
    plt.savefig(p, dpi=300 if p.suffix == ".png" else None, bbox_inches="tight")
  plt.close(fig)