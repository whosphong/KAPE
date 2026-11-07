from __future__ import annotations
import numpy as np


def reshape_2d(X: np.ndarray) -> np.ndarray:
  """ensures atleast 2D array is passed through NWE"""
  X = np.asarray(X, dtype=float)
  if X.ndim == 1:
    return X[:, None] # output shape (N, 1) if only 1 indep. variable passed through i.e. (N,)
  else:
    return X


class KAPE:
    """Kernel Adaptive Plasma Estimator (KAPE) model for regression with allowed uniform bandwidth or adaptive bandwidth.
    i am only taking in gaussian kernel can be edited wanted but speciifcally only gaussian for now. It is also assumed we had identiy matrix for the
    bandwidth as opposed to a covariance matrix to simplify inversion matrix"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        h: float | np.ndarray,
        Q0: float | None = None,
    ) -> None:

        self.X = reshape_2d(X).astype(np.float32, copy=False)
        self.y = np.asarray(y, dtype=float).ravel().astype(np.float32, copy=False)
        self.d = self.X.shape[1]  # Dimension d of features

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X and y dimension mismatch in sample size values, X shape is {self.X.shape}, y shape is {self.y.shape}")

        ########################
        n = self.X.shape[0]
        min_h = 1e-8  # Lowebound to make sure bandwidth is not too small to avoid numerical issues
        h = np.asarray(h, dtype=float).ravel()
        # Allow for a unfirom bandiwdth value for all training points (IS NOT ADAPTIVE AT ALL)
        if h.size == 1:
            val = max(min_h, float(h[0]))
            self.h_vec = np.full(n, val, dtype=np.float32)
        # if its the same length as the training points then we assume it is adaptive or whatever vector they inputted for h
        elif h.shape[0] == n:
            self.h_vec = np.maximum(min_h, np.nan_to_num(h, nan=min_h)).astype(np.float32)
        else:
            raise ValueError(f"h dimension mismatch with training points: expected 1 or {n} elements, got {h.shape[0]}")
        #######################

        self.h_sq = (self.h_vec**2).astype(np.float32)  # Variance term h_i^2 (h as standard deviation)
        self.log_h = np.log(self.h_vec)
        self.Q0 = Q0
        self.X_fit: np.ndarray | None = None
        self.y_fit: np.ndarray | None = None
        self.q_f: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self) -> KAPE:
        """fit the model to the trianing data definde in model initialization"""

        X_train = self.X
        y_train = self.y

        self.X_fit = X_train
        self.y_fit = y_train
        self.q_f = np.sum(self.X_fit * self.X_fit, axis=1)  # ||X||^2
        self.is_fitted = True
        return self

    def support_Q0(self, pct: float = 95.0) -> float:
        """Compute the support value for Q0 based on the training data via a percentile of the minimum distances between training points."""

        if not self.is_fitted:
            raise RuntimeError("We requre the model to be fitted first.")
        dist = (self.q_f[:, None]+ self.q_f[None, :]- 2.0 * (self.X_fit @ self.X_fit.T)) / self.h_sq[None, :]  # ||X_i - X_j||^2 / h_i^2 = (||X_i||^2 + ||X_j||^2 - 2 X_i^T X_j) / h_i^2
        dist = np.maximum(dist, 0.0)
        np.fill_diagonal(dist, np.inf)  # ensures that the minimum distance for each point is not itself, but the closest other point. This is important for computing the support value for Q0, which is based on the distances between training points.
        self.Q0 = float(np.percentile(dist.min(axis=1), pct))
        return self.Q0

    def predict(self, X_query: np.ndarray, prior: bool = False) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("We requre the model to be fitted first.")

        Xq = reshape_2d(X_query).astype(np.float32, copy=False)
        q_q = np.sum(Xq * Xq, axis=1)  # ||X_q||^2
        q_qf = Xq @ self.X_fit.T  # X_q^T * X_f
        Q = (q_q[:, None] + self.q_f[None, :] - 2.0 * q_qf) / self.h_sq[None, :]  # ||X_q - X_f||^2 / h_f^2 = (||X_q||^2 + ||X_f||^2 - 2 X_q^T X_f) / h_f^2
        logw = (-self.d * self.log_h[None, :] - 0.5 * Q)  # logw = -d*log(h_f) - 0.5*||X_q - X_f||^2 / h_f^2
        if (not prior) or (self.Q0 is None):
            m = logw.max(axis=1, keepdims=True)
            w = np.exp(logw - m) # trick to avoid numerical issues with large exponents, we subtract the maximum value of logw from each element of logw before exponentiating. This ensures that the largest value of logw is 0, and all other values are negative, which prevents overflow when exponentiating.
            return (w @ self.y_fit) / w.sum(axis=1)
        logw0 = -self.d * np.log(np.median(self.h_vec)) - 0.5 * self.Q0 # logw0 = -d*log(h_median) - 0.5Q0, where h_median is the median of the bandwidth vector h_vec, and Q0 is the support value for the prior. This term represents the contribution of the prior to the prediction, and is added to the weights of the training points when computing the prediction.
        m = np.maximum(logw.max(axis=1, keepdims=True), logw0)
        w = np.exp(logw - m)
        w0 = np.exp(logw0 - m)
        return (w @ self.y_fit) / (w.sum(axis=1, keepdims=True) + w0).ravel()
        

# class KAPE:
#   """LEGACY CODE"""

#   def __init__(
#       self,
#       X: np.ndarray,
#       y: np.ndarray,
#       h: float | np.ndarray,
#       Q0: float | None = None,
#   ) -> None:
#     self.X = reshape_2d(X).astype(np.float32, copy=False)
#     self.y = np.asarray(y, dtype=float).ravel().astype(np.float32, copy=False)
#     self.d = self.X.shape[1]

#     if self.X.shape[0] != self.y.shape[0]:
#       raise ValueError(
#           f"X and y dimension mismatch in sample size values, X shape is {self.X.shape}, y shape is {self.y.shape}"
#       )

#     n = self.X.shape[0]
#     min_h = 1e-8
#     h_arr = np.asarray(h, dtype=float).ravel()

#     if h_arr.size == 1:
#       val = max(min_h, float(h_arr[0]))
#       self.h_vec = np.full(n, val, dtype=np.float32)
#     elif h_arr.shape[0] == n:
#       self.h_vec = np.maximum(min_h, np.nan_to_num(h_arr, nan=min_h)).astype(
#           np.float32
#       )
#     else:
#       raise ValueError(
#           f"h dimension mismatch with training points: expected 1 or {n} elements, got {h_arr.shape[0]}"
#       )

#     self.h_sq = (self.h_vec**2).astype(np.float32)
#     self.log_h = np.log(self.h_vec)
#     self.Q0 = Q0
#     self.X_fit: np.ndarray | None = None
#     self.y_fit: np.ndarray | None = None
#     self.q_f: np.ndarray | None = None
#     self.is_fitted: bool = False

#   def fit(self) -> KAPE:
#     """fit the model to the training data defined in model initialization"""
#     self.X_fit = self.X
#     self.y_fit = self.y
#     self.q_f = np.sum(self.X_fit * self.X_fit, axis=1)
#     self.is_fitted = True
#     return self

#   def support_Q0(self, pct: float = 95.0) -> float:
#     """Compute the support value for Q0 based on the training data via a percentile of the minimum distances between training points."""
#     if not self.is_fitted or self.X_fit is None or self.q_f is None:
#       raise RuntimeError("We require the model to be fitted first.")

#     dist = (
#         self.q_f[:, None] + self.q_f[None, :] - 2.0 * (self.X_fit @ self.X_fit.T)
#     ) / self.h_vec[None, :]
#     dist = np.maximum(dist, 0.0)
#     np.fill_diagonal(dist, np.inf)
#     self.Q0 = float(np.percentile(dist.min(axis=1), pct))
#     return self.Q0

#   def predict(self, X_query: np.ndarray, prior: bool = False) -> np.ndarray:
#     if not self.is_fitted or self.X_fit is None or self.y_fit is None or self.q_f is None:
#       raise RuntimeError("We require the model to be fitted first.")

#     Xq = reshape_2d(X_query).astype(np.float32, copy=False)
#     q_q = np.sum(Xq * Xq, axis=1)
#     q_qf = Xq @ self.X_fit.T

#     Q = (q_q[:, None] + self.q_f[None, :] - 2.0 * q_qf) / self.h_vec[None, :]
#     logw = -1.5 * self.log_h[None, :] - 0.5 * Q

#     if (not prior) or (self.Q0 is None):
#       m = logw.max(axis=1, keepdims=True)
#       w = np.exp(logw - m)
#       return (w @ self.y_fit) / w.sum(axis=1)

#     logw0 = -1.5 * np.log(np.median(self.h_vec)) - 0.5 * self.Q0
#     m = np.maximum(logw.max(axis=1, keepdims=True), logw0)
#     w = np.exp(logw - m)
#     w0 = np.exp(logw0 - m)
#     return (w @ self.y_fit) / (w.sum(axis=1, keepdims=True) + w0).ravel()


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