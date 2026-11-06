from __future__ import annotations
import numpy as np


def reshape_2d(X: np.ndarray) -> np.ndarray:
  """Ensures input vectors are 2D column matrices."""
  X = np.asarray(X, dtype=float)
  return X[:, None] if X.ndim == 1 else X


class NAE:
  """Nadaraya-Watson Estimator with optional prior regularizer scale (Q0)."""

  def __init__(
      self, h_vec: np.ndarray, Q0: float | None = None
  ) -> None:
    h_arr = np.asarray(h_vec, dtype=float).ravel()
    min_eps = 1e-8
    self.h_vec = np.maximum(min_eps, np.nan_to_num(h_arr, nan=min_eps)).astype(
        np.float32
    )
    self.log_h = np.log(self.h_vec)
    self.Q0 = Q0
    self.X_fit: np.ndarray | None = None
    self.y_fit: np.ndarray | None = None
    self.q_f: np.ndarray | None = None
    self.is_fitted: bool = False

  def fit(self, X: np.ndarray, y: np.ndarray) -> NAE:
    X_fit = reshape_2d(X).astype(np.float32, copy=False)
    y_fit = np.asarray(y, dtype=float).ravel().astype(np.float32, copy=False)

    if X_fit.shape[0] != y_fit.shape[0]:
      raise ValueError("X and y dimension mismatch")
    if self.h_vec.shape[0] != X_fit.shape[0]:
      raise ValueError("h_vec dimension mismatch with training points")

    self.X_fit = X_fit
    self.y_fit = y_fit
    self.q_f = np.sum(X_fit * X_fit, axis=1)
    self.is_fitted = True
    return self

  def support_Q0(self, pct: float = 95.0) -> float:
    if not self.is_fitted or self.X_fit is None or self.q_f is None:
      raise RuntimeError("Call fit first")
    Qtt = (
        self.q_f[:, None]
        + self.q_f[None, :]
        - 2.0 * (self.X_fit @ self.X_fit.T)
    ) / self.h_vec[None, :]
    Qtt = np.maximum(Qtt, 0.0)
    np.fill_diagonal(Qtt, np.inf)
    return float(np.percentile(Qtt.min(axis=1), pct))

  def predict(self, X_query: np.ndarray, use_prior: bool = False) -> np.ndarray:
    if (
        not self.is_fitted
        or self.X_fit is None
        or self.y_fit is None
        or self.q_f is None
    ):
      raise RuntimeError("Call fit first")

    Xq = reshape_2d(X_query).astype(np.float32, copy=False)
    q_q = np.sum(Xq * Xq, axis=1)
    q_qf = Xq @ self.X_fit.T

    Q = (q_q[:, None] + self.q_f[None, :] - 2.0 * q_qf) / self.h_vec[None, :]
    logw = -1.5 * self.log_h[None, :] - 0.5 * Q

    if (not use_prior) or (self.Q0 is None):
      m = logw.max(axis=1, keepdims=True)
      w = np.exp(logw - m)
      return (w @ self.y_fit) / w.sum(axis=1)

    logw0 = -1.5 * np.log(np.median(self.h_vec)) - 0.5 * self.Q0
    m = np.maximum(logw.max(axis=1, keepdims=True), logw0)
    w = np.exp(logw - m)
    w0 = np.exp(logw0 - m)
    return (w @ self.y_fit) / (w.sum(axis=1, keepdims=True) + w0).ravel()


def get_domain_bounds(
    X: np.ndarray,
    domain_mode: str = "buffer",
    buffer_r: float = 5.0,
    buffer_angle_rad: float = np.radians(10.0),
) -> tuple[float, float, float, float, float, float]:
  vx, vy, vz = X[:, 0], X[:, 1], X[:, 2]
  r = np.sqrt(vx**2 + vy**2 + vz**2)
  theta = np.arctan2(vy, vx)
  phi = np.arcsin(np.clip(vz / np.where(r == 0, 1.0, r), -1.0, 1.0))

  if domain_mode == "fov":
    r_min, r_max = r.min(), r.max()
    theta_min, theta_max = theta.min(), theta.max()
    phi_min, phi_max = phi.min(), phi.max()
  elif domain_mode == "buffer":
    r_min = max(0.0, r.min() - buffer_r)
    r_max = r.max() + buffer_r
    theta_min = max(-np.pi, theta.min() - buffer_angle_rad)
    theta_max = min(np.pi, theta.max() + buffer_angle_rad)
    phi_min = max(-np.pi / 2, phi.min() - buffer_angle_rad)
    phi_max = min(np.pi / 2, phi.max() + buffer_angle_rad)
  else:  # 'full'
    r_min, r_max = 0.0, r.max() * 1.5
    theta_min, theta_max = -np.pi, np.pi
    phi_min, phi_max = -np.pi / 2, np.pi / 2

  return r_min, r_max, theta_min, theta_max, phi_min, phi_max


def in_spherical_domain(
    Xq_car: np.ndarray,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    domain_mode: str = "buffer",
) -> np.ndarray:
  if domain_mode == "full":
    return np.ones(len(Xq_car), dtype=bool)

  vx_, vy_, vz_ = Xq_car[:, 0], Xq_car[:, 1], Xq_car[:, 2]
  r_q = np.sqrt(vx_**2 + vy_**2 + vz_**2)

  valid_r = (r_q >= r_min) & (r_q <= r_max)
  inside = np.zeros(len(Xq_car), dtype=bool)
  if not np.any(valid_r):
    return inside

  vx_v, vy_v, vz_v, r_v = (
      vx_[valid_r],
      vy_[valid_r],
      vz_[valid_r],
      r_q[valid_r],
  )
  theta_q = np.arctan2(vy_v, vx_v)
  phi_q = np.arcsin(np.clip(vz_v / np.maximum(r_v, 1e-12), -1.0, 1.0))

  valid_angles = (
      (theta_q >= theta_min)
      & (theta_q <= theta_max)
      & (phi_q >= phi_min)
      & (phi_q <= phi_max)
  )

  inside[valid_r] = valid_angles
  return inside