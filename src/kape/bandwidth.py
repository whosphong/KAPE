from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from .core import KAPE


def compute_bandwidth_from_array(X: np.ndarray) -> np.ndarray:

  """
  Computes the bandwidth h for each point in X based on the maximum distance to its nearest neighbors in spherical coordinates.
  this should not change in terms of x,y,z coordinates of an esa output.
  """

  vx, vy, vz = X[:, 0], X[:, 1], X[:, 2]

  # rounding because of intial transformation error from ESA data to cartesian into spherical coordinates.
  
  r = np.sqrt(vx**2 + vy**2 + vz**2)
  theta = np.round(np.arctan2(vy, vx), 6)
  phi = np.round(
      np.arcsin(np.clip(vz / np.where(r == 0, 1.0, r), -1.0, 1.0)), 5
  )

  # need unique values as computing euclidean distance between points in spherical coordinates, so need unique values to avoid duplicates.

  unique_theta = np.unique(theta)
  unique_phi = np.unique(phi)

  N_th = len(unique_theta) # should be 8
  N_ph = len(unique_phi) # should be 8

  th_idx_pts = np.searchsorted(unique_theta, theta) # points where sample points are located in the unique theta array the indexes they would live in
                                                    # in the sorted unqiue theta array. find what value theta is closest to and return the index of that value in the unique theta array. this is used to find the neighbors of each point in the spherical domain.
  ph_idx_pts = np.searchsorted(unique_phi, phi)

  th_all = np.repeat(unique_theta, N_ph) # create a grid of all possible theta and phi combinations in the spherical domain. this is used to compute the chord distance between points in the spherical domain.
  ph_all = np.tile(unique_phi, N_th)

  sin_ph_all = np.sin(ph_all)
  cos_ph_all = np.cos(ph_all)

  # Edge Case handled: Float underflow causing NaN is clamped at 0.0
  chord = np.sqrt(np.maximum(0.0, 2 - 2 * (sin_ph_all[:, None] * sin_ph_all[None, :] + cos_ph_all[:, None] * cos_ph_all[None, :] * np.cos(th_all[:, None] - th_all[None, :]))))

  chord_grid = chord.reshape(N_th, N_ph, N_th, N_ph)
  h = np.empty(len(r))

  for k in range(len(r)):
    i, j = th_idx_pts[k], ph_idx_pts[k]
    r_k = r[k]

    same_ray_mask = (th_idx_pts == i) & (ph_idx_pts == j)
    same_ray_mask[k] = False
    r_same = r[same_ray_mask]

    # Edge Case handled: Handles missing inner/outer neighbors at r_min and r_max boundaries
    r_max_dist = 0.0
    if len(r_same) > 0:
      diffs = r_same - r_k
      outer = diffs[diffs > 0]
      inner = np.abs(diffs[diffs < 0])
      
      r_adjacent_dists = []
      if len(outer) > 0: r_adjacent_dists.append(np.min(outer))
      if len(inner) > 0: r_adjacent_dists.append(np.min(inner))
      
      if r_adjacent_dists:
        r_max_dist = max(r_adjacent_dists)

    # Edge Case handled: Safely ignores non-existent neighbors at bounded FOV limits
    neighbors = []
    for ni, nj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
      if 0 <= ni < N_th and 0 <= nj < N_ph:
        neighbors.append((ni, nj))

    cross_dists = [r_k * chord_grid[i, j, ni, nj] for ni, nj in neighbors]
    cross_max = np.max(cross_dists) if cross_dists else 0.0

    h[k] = max(r_max_dist, cross_max)

  return h



def opt_alph(
    X: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    alphas: np.ndarray | None = None,
    n_splits: int = 25,
    random_state: int = 42,
    show_progress: bool = True,
) -> tuple[float, dict[float, float], dict[float, float]]:

  """
  Simple Kfold cross validaition to find the optimal alpha for h = alpha * D
  """

  X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
  if alphas is None: alphas = np.linspace(0.1, 2, 60)

  kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
  results_mean, results_std = {}, {}

  alpha_iter = tqdm(alphas, desc="k-CV Alpha Optimization", leave=False, disable=not show_progress)

  for alpha in alpha_iter:
    fold_mses = []
    for train_idx, val_idx in kf.split(X):
      model = KAPE(X[train_idx], y[train_idx], h[train_idx] * alpha)
      y_pred = model.fit().predict(X[val_idx])
      fold_mses.append(np.mean((y_pred - y[val_idx]) ** 2))

    results_mean[alpha] = float(np.mean(fold_mses))
    results_std[alpha] = float(np.std(fold_mses))

  best_alpha = min(results_mean, key=results_mean.get)
  return best_alpha, results_mean, results_std