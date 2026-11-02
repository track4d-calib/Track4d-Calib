from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _as_np64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _torch_available() -> bool:
    try:
        import torch  # noqa: WPS433

        return True
    except Exception:  # pragma: no cover
        return False


def rodrigues_torch(rvec, *, eps: float = 1e-8):
    """
    Differentiable Rodrigues formula (axis-angle -> rotation matrix).

    Args:
        rvec: (..., 3) axis-angle tensor
    Returns:
        (..., 3, 3) rotation matrices
    """
    import torch  # noqa: WPS433

    r = rvec
    # NOTE:
    # torch.linalg.norm has undefined gradient at exactly zero. In practice, many
    # cameras initialize with identity rotation (rvec==0), which can produce NaN
    # gradients and poison optimization. Build theta from squared norm with a
    # floor so gradients remain finite at/near zero.
    theta2 = torch.sum(r * r, dim=-1, keepdim=True)  # (..., 1)
    theta2_safe = torch.clamp(theta2, min=float(eps))
    theta = torch.sqrt(theta2_safe)
    # Use the unnormalized rotation vector for the stable Rodrigues formula:
    # R = I + (sinθ/θ) [r]_x + ((1-cosθ)/θ^2) [r]_x^2
    k = r
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zeros, -kz, ky], dim=-1),
            torch.stack([kz, zeros, -kx], dim=-1),
            torch.stack([-ky, kx, zeros], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)

    # Use stable sinc-like factors (with small-angle Taylor expansions).
    small = theta2 < 1e-8
    A_taylor = 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0
    B_taylor = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0
    A = torch.where(small, A_taylor, torch.sin(theta) / theta)
    B = torch.where(small, B_taylor, (1.0 - torch.cos(theta)) / theta2_safe)

    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    while eye.ndim < K.ndim:  # broadcast to batch dims
        eye = eye.unsqueeze(0)

    return eye + A[..., None] * K + B[..., None] * (K @ K)


def project_points_torch(
    X_world: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Project 3D points with OpenCV 5-parameter distortion model.

    This intentionally matches cv2.projectPoints (k1,k2,p1,p2,k3).

    Args:
        X_world: (N, 3) world points.
        fx,fy,cx,cy: intrinsics.
        dist: (5,) distortion coefficients.
        R: (3,3) world->camera rotation.
        t: (3,) world->camera translation.
    Returns:
        (N, 2) pixel coordinates.
    """
    if not _torch_available():  # pragma: no cover
        raise RuntimeError("project_points_torch requires PyTorch to be installed")

    import torch  # noqa: WPS433

    X = torch.tensor(_as_np64(X_world), dtype=torch.float64, device=device)
    R_t = torch.tensor(_as_np64(R), dtype=torch.float64, device=device)
    t_t = torch.tensor(_as_np64(t).reshape(3), dtype=torch.float64, device=device)
    D = torch.tensor(_as_np64(dist).reshape(-1), dtype=torch.float64, device=device)

    Xc = (R_t @ X.T).T + t_t[None, :]
    x = Xc[:, 0] / Xc[:, 2].clamp_min(1e-9)
    y = Xc[:, 1] / Xc[:, 2].clamp_min(1e-9)

    k1, k2, p1, p2, k3 = D[:5]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_rad = x * radial
    y_rad = y * radial

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    xd = x_rad + x_tan
    yd = y_rad + y_tan

    u = float(fx) * xd + float(cx)
    v = float(fy) * yd + float(cy)
    uv = torch.stack([u, v], dim=-1)
    return uv.detach().cpu().numpy().astype(np.float64)


def optimize_camera_parameters_torch(
    *,
    X_world: np.ndarray,
    uv_obs: np.ndarray,
    K_init: np.ndarray,
    D_init: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    image_hw: Tuple[int, int],
    device: str = "cpu",
    max_steps: int = 200,
    batch_size: int = 8192,
    lr: float = 5e-2,
    huber_delta: float = 3.0,
    reg_weight: float = 1e-2,
    optimize_intrinsics: bool = True,
    optimize_distortion: bool = True,
    optimize_extrinsics: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Optimize a single camera (K + D + extrinsic) from 3D-2D correspondences.

    This is intended as an inner-loop optimizer for multi-view video calibration refinement.

    Returns:
        dict with keys: K, D, R, t
    """
    if not _torch_available():  # pragma: no cover
        raise RuntimeError("optimize_camera_parameters_torch requires PyTorch to be installed")

    import cv2  # noqa: WPS433
    import torch  # noqa: WPS433

    X_world = _as_np64(X_world)
    uv_obs = _as_np64(uv_obs)
    if X_world.ndim != 2 or X_world.shape[1] != 3:
        raise ValueError("X_world must have shape (N,3)")
    if uv_obs.ndim != 2 or uv_obs.shape[1] != 2:
        raise ValueError("uv_obs must have shape (N,2)")
    if X_world.shape[0] != uv_obs.shape[0]:
        raise ValueError("X_world and uv_obs must have same length")

    H, W = int(image_hw[0]), int(image_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError("image_hw must be positive")

    K0 = _as_np64(K_init)
    D0 = _as_np64(D_init).reshape(-1)
    R0 = _as_np64(R_init)
    t0 = _as_np64(t_init).reshape(3)

    fx0, fy0, cx0, cy0 = float(K0[0, 0]), float(K0[1, 1]), float(K0[0, 2]), float(K0[1, 2])
    rvec0, _ = cv2.Rodrigues(R0)
    rvec0 = rvec0.reshape(3)

    # Parameters (use log for focal lengths to enforce positivity).
    log_fx_t = torch.tensor(np.log(max(fx0, 1e-6)), dtype=torch.float64, device=device)
    log_fy_t = torch.tensor(np.log(max(fy0, 1e-6)), dtype=torch.float64, device=device)
    cx_t = torch.tensor(cx0, dtype=torch.float64, device=device)
    cy_t = torch.tensor(cy0, dtype=torch.float64, device=device)
    dist_t = torch.tensor(D0[:5], dtype=torch.float64, device=device)
    rvec_t = torch.tensor(rvec0, dtype=torch.float64, device=device)
    tvec_t = torch.tensor(t0, dtype=torch.float64, device=device)

    log_fx = torch.nn.Parameter(log_fx_t) if bool(optimize_intrinsics) else log_fx_t
    log_fy = torch.nn.Parameter(log_fy_t) if bool(optimize_intrinsics) else log_fy_t
    cx = torch.nn.Parameter(cx_t) if bool(optimize_intrinsics) else cx_t
    cy = torch.nn.Parameter(cy_t) if bool(optimize_intrinsics) else cy_t
    dist = torch.nn.Parameter(dist_t) if bool(optimize_distortion) else dist_t
    rvec = torch.nn.Parameter(rvec_t) if bool(optimize_extrinsics) else rvec_t
    tvec = torch.nn.Parameter(tvec_t) if bool(optimize_extrinsics) else tvec_t

    params = []
    if bool(optimize_intrinsics):
        params.extend([log_fx, log_fy, cx, cy])
    if bool(optimize_distortion):
        params.append(dist)
    if bool(optimize_extrinsics):
        params.extend([rvec, tvec])

    X_t = torch.tensor(X_world, dtype=torch.float64, device=device)
    uv_t = torch.tensor(uv_obs, dtype=torch.float64, device=device)

    # Store init values for regularization.
    init = {
        "log_fx": float(log_fx_t.detach().cpu().numpy()),
        "log_fy": float(log_fy_t.detach().cpu().numpy()),
        "cx": float(cx_t.detach().cpu().numpy()),
        "cy": float(cy_t.detach().cpu().numpy()),
        "dist": dist_t.detach().clone(),
        "rvec": rvec_t.detach().clone(),
        "tvec": tvec_t.detach().clone(),
    }

    def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
        ax = torch.abs(x)
        quad = torch.minimum(ax, torch.tensor(delta, dtype=ax.dtype, device=ax.device))
        lin = ax - quad
        return 0.5 * quad * quad + delta * lin

    opt = torch.optim.Adam(params, lr=float(lr))

    n = int(X_world.shape[0])
    batch_size_eff = min(int(batch_size), n)
    if batch_size_eff <= 0:
        raise ValueError("No observations to optimize")

    unstable_optimization = False
    for _ in range(int(max_steps)):
        # Stochastic minibatch helps keep runtime bounded for large sequences.
        if batch_size_eff == n:
            idx = None
            Xb = X_t
            ub = uv_t
        else:
            idx = torch.randint(0, n, size=(batch_size_eff,), device=device)
            Xb = X_t[idx]
            ub = uv_t[idx]

        fx = torch.exp(log_fx)
        fy = torch.exp(log_fy)

        R = rodrigues_torch(rvec)
        Xc = (R @ Xb.T).T + tvec[None, :]
        x = Xc[:, 0] / Xc[:, 2].clamp_min(1e-9)
        y = Xc[:, 1] / Xc[:, 2].clamp_min(1e-9)

        k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_rad = x * radial
        y_rad = y * radial
        x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        xd = x_rad + x_tan
        yd = y_rad + y_tan

        u_pred = fx * xd + cx
        v_pred = fy * yd + cy
        uv_pred = torch.stack([u_pred, v_pred], dim=-1)

        resid = uv_pred - ub
        data_loss = huber(resid, float(huber_delta)).mean()

        # Regularize towards initial calibration to avoid degeneracies (esp. scale).
        reg = torch.tensor(0.0, dtype=torch.float64, device=device)
        if bool(optimize_intrinsics):
            reg = reg + (log_fx - init["log_fx"]) ** 2
            reg = reg + (log_fy - init["log_fy"]) ** 2
            reg = reg + ((cx - init["cx"]) / max(W, 1)) ** 2
            reg = reg + ((cy - init["cy"]) / max(H, 1)) ** 2
        if bool(optimize_distortion):
            reg = reg + ((dist - init["dist"]).pow(2).mean())
        if bool(optimize_extrinsics):
            reg = reg + ((rvec - init["rvec"]).pow(2).mean())
            reg = reg + ((tvec - init["tvec"]).pow(2).mean())

        loss = data_loss + float(reg_weight) * reg
        if not bool(torch.isfinite(loss).item()):
            unstable_optimization = True
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Keep parameters in a sane range to avoid optimizer explosions.
        with torch.no_grad():
            if bool(optimize_intrinsics):
                log_fx.clamp_(np.log(1.0), np.log(1e5))
                log_fy.clamp_(np.log(1.0), np.log(1e5))
                cx.clamp_(0.0, float(W))
                cy.clamp_(0.0, float(H))
            if bool(optimize_distortion):
                # OpenCV distortion is typically small; clamp aggressively.
                dist[0].clamp_(-1.0, 1.0)  # k1
                dist[1].clamp_(-1.0, 1.0)  # k2
                dist[2].clamp_(-0.1, 0.1)  # p1
                dist[3].clamp_(-0.1, 0.1)  # p2
                dist[4].clamp_(-1.0, 1.0)  # k3
            if bool(optimize_extrinsics):
                rvec.clamp_(-3.14, 3.14)
                tvec.clamp_(-10.0, 10.0)

            if not bool(
                torch.isfinite(log_fx).all()
                and torch.isfinite(log_fy).all()
                and torch.isfinite(cx).all()
                and torch.isfinite(cy).all()
                and torch.isfinite(dist).all()
                and torch.isfinite(rvec).all()
                and torch.isfinite(tvec).all()
            ):
                unstable_optimization = True
                break

    if unstable_optimization:
        d0 = np.zeros((5,), dtype=np.float64)
        d0[: min(5, D0.shape[0])] = D0[:5]
        return {
            "K": np.asarray(K0, dtype=np.float64),
            "D": d0,
            "R": np.asarray(R0, dtype=np.float64),
            "t": np.asarray(t0, dtype=np.float64).reshape(3),
        }

    # Convert back to numpy.
    fx1 = float(torch.exp(log_fx).detach().cpu().numpy())
    fy1 = float(torch.exp(log_fy).detach().cpu().numpy())
    cx1 = float(cx.detach().cpu().numpy())
    cy1 = float(cy.detach().cpu().numpy())
    D1 = dist.detach().cpu().numpy().reshape(5).astype(np.float64)
    rvec1 = rvec.detach().cpu().numpy().reshape(3).astype(np.float64)
    t1 = tvec.detach().cpu().numpy().reshape(3).astype(np.float64)

    if not (
        np.isfinite(fx1)
        and np.isfinite(fy1)
        and np.isfinite(cx1)
        and np.isfinite(cy1)
        and np.isfinite(D1).all()
        and np.isfinite(rvec1).all()
        and np.isfinite(t1).all()
    ):
        d0 = np.zeros((5,), dtype=np.float64)
        d0[: min(5, D0.shape[0])] = D0[:5]
        return {
            "K": np.asarray(K0, dtype=np.float64),
            "D": d0,
            "R": np.asarray(R0, dtype=np.float64),
            "t": np.asarray(t0, dtype=np.float64).reshape(3),
        }

    R1, _ = cv2.Rodrigues(rvec1)

    K1 = np.array([[fx1, 0.0, cx1], [0.0, fy1, cy1], [0.0, 0.0, 1.0]], dtype=np.float64)
    return {"K": K1, "D": D1, "R": np.asarray(R1, dtype=np.float64), "t": t1}
