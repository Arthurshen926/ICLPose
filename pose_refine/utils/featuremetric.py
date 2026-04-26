"""
Featuremetric Direct Alignment (FDA)
=====================================
纯几何方法：通过渲染特征的空间梯度+ Image Jacobian 直接推导位姿修正

原理：
  对于特征图 f(u,v)，当相机移动 δξ 时，固定像素处的特征变化为：
    Δf(u,v) = -(∂f/∂u · Δu + ∂f/∂v · Δv)
  其中 (Δu, Δv) = Image Jacobian · δξ

  最小化 ||f_query - f_rendered(ξ + δξ)||² 的 Gauss-Newton 更新：
    (J^T J + λI) δξ = -J^T r
  其中 r = f_query - f_rendered, J = 特征 Jacobian

核心优势：
  - 无需训练！纯几何推导
  - 利用全部特征维度（640+768=1408维 × 1610 像素 = 226万约束求解 6 未知数）
  - 精确的梯度方向，不依赖神经网络学习

等价于：
  - Lucas-Kanade 光流的高维特征版本
  - DSO (Direct Sparse Odometry) 的特征度量版本
  - iNeRF 但用解析 Jacobian 代替数值求导

参考：
  - Lucas & Kanade (1981): Iterative Image Registration
  - Engel et al. (2018): DSO — Direct Sparse Odometry
  - Yen-Chen et al. (2021): iNeRF
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def compute_image_jacobian(
    depth: torch.Tensor,
    intrinsics: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 Image Jacobian: 像素位移关于相机运动 ξ 的 Jacobian

    对于像素 (u,v) 在深度 Z 处，相机运动 ξ=[tx,ty,tz,ωx,ωy,ωz] 产生的像素位移：
      Δu = [fx/Z, 0, -fx·x/Z, -fx·xy, fx(1+x²), -fx·y] · ξ
      Δv = [0, fy/Z, -fy·y/Z, -fy(1+y²), fy·xy, fy·x] · ξ

    Args:
        depth: (B, H, W) 深度图
        intrinsics: {'fx', 'fy', 'cx', 'cy'}

    Returns:
        Ju: (B, N, 6) u方向的 Jacobian
        Jv: (B, N, 6) v方向的 Jacobian
        valid: (B, N) 有效像素 mask
    """
    B, H, W = depth.shape
    N = H * W
    device = depth.device

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    x = (u_coords - cx) / fx  # (H, W)
    y = (v_coords - cy) / fy
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    y = y.unsqueeze(0).expand(B, -1, -1)

    Z = depth.clamp(min=0.05)
    inv_Z = 1.0 / Z
    valid = (depth > 0.05).reshape(B, N)

    x2, y2, xy = x * x, y * y, x * y

    # Ju: (B, H, W, 6)
    Ju = torch.stack([
        fx * inv_Z,                     # tx
        torch.zeros_like(x),            # ty
        -fx * x * inv_Z,                # tz
        -fx * xy,                        # ωx
        fx * (1.0 + x2),                # ωy
        -fx * y,                         # ωz
    ], dim=-1)

    Jv = torch.stack([
        torch.zeros_like(y),            # tx
        fy * inv_Z,                     # ty
        -fy * y * inv_Z,                # tz
        -fy * (1.0 + y2),               # ωx
        fy * xy,                         # ωy
        fy * x,                          # ωz
    ], dim=-1)

    return Ju.reshape(B, N, 6), Jv.reshape(B, N, 6), valid


def compute_spatial_gradient(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算特征图的空间梯度（中心差分）

    Args:
        features: (B, D, H, W)

    Returns:
        grad_u: (B, D, H, W) u(水平)方向梯度
        grad_v: (B, D, H, W) v(垂直)方向梯度
    """
    # 中心差分，边界用前向/后向差分
    # u方向 (水平)
    grad_u = torch.zeros_like(features)
    grad_u[:, :, :, 1:-1] = (features[:, :, :, 2:] - features[:, :, :, :-2]) / 2.0
    grad_u[:, :, :, 0] = features[:, :, :, 1] - features[:, :, :, 0]
    grad_u[:, :, :, -1] = features[:, :, :, -1] - features[:, :, :, -2]

    # v方向 (垂直)
    grad_v = torch.zeros_like(features)
    grad_v[:, :, 1:-1, :] = (features[:, :, 2:, :] - features[:, :, :-2, :]) / 2.0
    grad_v[:, :, 0, :] = features[:, :, 1, :] - features[:, :, 0, :]
    grad_v[:, :, -1, :] = features[:, :, -1, :] - features[:, :, -2, :]

    return grad_u, grad_v


def featuremetric_gauss_newton_step(
    query_feats: torch.Tensor,
    rendered_feats: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: Dict[str, float],
    damping: float = 1e-4,
) -> torch.Tensor:
    """
    一步 Gauss-Newton 特征度量对准

    最小化 ||f_query - f_rendered(ξ + δξ)||²

    利用高效的正规方程：不需要显式构建 (D*N, 6) 的 Jacobian 矩阵
    只需 O(N) 的中间变量

    Args:
        query_feats: (B, D, H, W) 查询图像特征（L2 归一化后）
        rendered_feats: (B, D, H, W) 渲染特征（L2 归一化后）
        depth: (B, H, W) 渲染深度图
        intrinsics: 相机内参
        damping: LM 阻尼因子

    Returns:
        delta_xi: (B, 6) se(3) 位姿修正量
    """
    B, D, H, W = query_feats.shape
    N = H * W

    # 1. 特征残差
    residual = query_feats - rendered_feats  # (B, D, H, W)

    # 2. 渲染特征的空间梯度
    grad_u, grad_v = compute_spatial_gradient(rendered_feats)  # (B, D, H, W) each

    # 3. Image Jacobian
    Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)  # (B, N, 6), (B, N, 6), (B, N)

    # 4. 高效正规方程（不显式构建大 Jacobian）
    # 需要的中间量：
    #   A[n] = Σ_d (∂f_d/∂u[n])² — 水平梯度能量
    #   B_[n] = Σ_d (∂f_d/∂v[n])² — 垂直梯度能量
    #   C[n] = Σ_d ∂f_d/∂u[n] · ∂f_d/∂v[n] — 交叉项
    #   Ru[n] = Σ_d ∂f_d/∂u[n] · r_d[n] — 水平 Jacobian-residual 乘积
    #   Rv[n] = Σ_d ∂f_d/∂v[n] · r_d[n] — 垂直 Jacobian-residual 乘积

    gu = grad_u.reshape(B, D, N)      # (B, D, N)
    gv = grad_v.reshape(B, D, N)      # (B, D, N)
    r = residual.reshape(B, D, N)     # (B, D, N)

    # 对特征维度求和
    A = (gu * gu).sum(dim=1)           # (B, N) — Σ_d (∂f_d/∂u)²
    B_ = (gv * gv).sum(dim=1)         # (B, N)
    C = (gu * gv).sum(dim=1)          # (B, N)
    Ru = (gu * r).sum(dim=1)          # (B, N) — Σ_d ∂f_d/∂u · r_d
    Rv = (gv * r).sum(dim=1)          # (B, N)

    # 应用 valid mask
    A = A * valid.float()
    B_ = B_ * valid.float()
    C = C * valid.float()
    Ru = Ru * valid.float()
    Rv = Rv * valid.float()

    # J^T J = Ju^T diag(A) Ju + Ju^T diag(C) Jv + Jv^T diag(C) Ju + Jv^T diag(B_) Jv
    Ju_A = Ju * A.unsqueeze(-1)      # (B, N, 6)
    Ju_C = Ju * C.unsqueeze(-1)
    Jv_C = Jv * C.unsqueeze(-1)
    Jv_B = Jv * B_.unsqueeze(-1)

    JtJ = (torch.bmm(Ju_A.transpose(1, 2), Ju) +
           torch.bmm(Ju_C.transpose(1, 2), Jv) +
           torch.bmm(Jv_C.transpose(1, 2), Ju) +
           torch.bmm(Jv_B.transpose(1, 2), Jv))     # (B, 6, 6)

    # J^T r = -(Ju^T Ru + Jv^T Rv)  (负号因为特征 Jacobian 中固有的负号)
    Ju_Ru = Ju * Ru.unsqueeze(-1)    # (B, N, 6)
    Jv_Rv = Jv * Rv.unsqueeze(-1)

    JtR = -(torch.bmm(Ju_Ru.transpose(1, 2), torch.ones(B, N, 1, device=Ju.device)) +
            torch.bmm(Jv_Rv.transpose(1, 2), torch.ones(B, N, 1, device=Jv.device)))

    # 更高效: JtR = -(Ju^T @ Ru_col + Jv^T @ Rv_col)
    JtR = -(torch.bmm(Ju.transpose(1, 2), Ru.unsqueeze(-1)) +
            torch.bmm(Jv.transpose(1, 2), Rv.unsqueeze(-1)))   # (B, 6, 1)

    # 5. LM 阻尼 + 求解
    damping_mat = damping * torch.eye(6, device=JtJ.device).unsqueeze(0)
    delta_xi = torch.linalg.solve(JtJ + damping_mat, JtR).squeeze(-1)  # (B, 6)

    return delta_xi


class FeaturemetricAligner:
    """
    特征度量直接对准器

    迭代地：
    1. 用当前位姿渲染特征图
    2. 计算特征残差 + 空间梯度 + Image Jacobian
    3. Gauss-Newton 更新位姿
    4. 重复

    无需训练！纯几何方法。
    """

    def __init__(
        self,
        renderer,
        intrinsics: Dict[str, float],
        scale_names: List[str],
        damping: float = 1e-2,
        damping_decrease: float = 1/3,
        damping_increase: float = 3.0,
        damping_min: float = 1e-6,
        damping_max: float = 1e4,
        max_iters: int = 30,
        convergence_thresh: float = 1e-5,
        rel_convergence_thresh: float = 0.005,
        rel_convergence_patience: int = 3,
        use_rendered_depth: bool = True,
    ):
        """
        Args:
            renderer: MultiScaleRenderer
            intrinsics: 相机内参 (渲染分辨率下)
            scale_names: 使用的尺度名称列表
            damping: 初始 LM 阻尼
            damping_decrease: 步成功时阻尼缩减因子
            damping_increase: 步失败时阻尼增长因子
            damping_min/max: 阻尼范围
            max_iters: 最大迭代次数
            convergence_thresh: 步长阈值 (弧度)
            rel_convergence_thresh: 相对残差改进阈值 (e.g. 0.005 = 0.5%)
            rel_convergence_patience: 连续多少次低于阈值则停止
            use_rendered_depth: 是否用渲染深度代替GT深度计算Jacobian
        """
        self.renderer = renderer
        self.intrinsics = intrinsics
        self.scale_names = scale_names
        self.damping_init = damping
        self.damping_decrease = damping_decrease
        self.damping_increase = damping_increase
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.max_iters = max_iters
        self.convergence_thresh = convergence_thresh
        self.rel_convergence_thresh = rel_convergence_thresh
        self.rel_convergence_patience = rel_convergence_patience
        self.use_rendered_depth = use_rendered_depth

    def _compute_residual_and_normal_eq(
        self,
        query_feats: Dict[str, torch.Tensor],
        rendered_all: Dict[str, torch.Tensor],
        depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """计算总残差和正规方程 (J^T J, J^T r)"""
        B = depth.shape[0]
        device = depth.device
        total_JtJ = torch.zeros(B, 6, 6, device=device)
        total_JtR = torch.zeros(B, 6, 1, device=device)
        total_residual = 0.0
        total_pixels = 0

        Ju, Jv, valid = compute_image_jacobian(depth, self.intrinsics)
        N = Ju.shape[1]

        for scale in self.scale_names:
            if scale not in rendered_all or scale not in query_feats:
                continue

            q = F.normalize(query_feats[scale], p=2, dim=1)
            r_feat = rendered_all[scale]  # 已 L2 归一化

            D = q.shape[1]
            residual = q - r_feat  # (B, D, H, W)

            # 空间梯度
            grad_u, grad_v = compute_spatial_gradient(r_feat)

            # 累加正规方程
            gu = grad_u.reshape(B, D, N)
            gv = grad_v.reshape(B, D, N)
            res = residual.reshape(B, D, N)

            A = (gu * gu).sum(1) * valid.float()
            B_ = (gv * gv).sum(1) * valid.float()
            C = (gu * gv).sum(1) * valid.float()
            Ru = (gu * res).sum(1) * valid.float()
            Rv = (gv * res).sum(1) * valid.float()

            JtJ = (torch.bmm((Ju * A.unsqueeze(-1)).transpose(1, 2), Ju) +
                   torch.bmm((Ju * C.unsqueeze(-1)).transpose(1, 2), Jv) +
                   torch.bmm((Jv * C.unsqueeze(-1)).transpose(1, 2), Ju) +
                   torch.bmm((Jv * B_.unsqueeze(-1)).transpose(1, 2), Jv))

            JtR = -(torch.bmm(Ju.transpose(1, 2), Ru.unsqueeze(-1)) +
                    torch.bmm(Jv.transpose(1, 2), Rv.unsqueeze(-1)))

            total_JtJ += JtJ
            total_JtR += JtR
            total_residual += (res ** 2).sum().item()
            total_pixels += valid.sum().item() * D

        avg_residual = total_residual / max(total_pixels, 1)
        return total_JtJ, total_JtR, avg_residual, total_pixels

    @torch.no_grad()
    def align(
        self,
        query_feats: Dict[str, torch.Tensor],
        initial_pose: torch.Tensor,
        depth_for_jac: torch.Tensor,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """
        Levenberg-Marquardt 特征度量对准

        关键改进：
        - 自适应阻尼：步成功(残差降低)时减小阻尼，步失败时增大阻尼并回退
        - 用渲染深度计算 Jacobian (可选)
        - 提前停止：连续多次失败则终止

        Args:
            query_feats: {scale_name: (B, D, H, W)} 查询特征
            initial_pose: (B, 4, 4) 初始位姿 (w2c)
            depth_for_jac: (B, H, W) 备用深度图 (当不使用渲染深度时)
            verbose: 是否打印迭代信息

        Returns:
            dict with 'final_pose', 'best_pose', 'residuals', 'num_iters', 'converged'
        """
        from pose_refine.utils.lie_algebra import se3_exp

        B = initial_pose.shape[0]
        device = initial_pose.device
        current_pose = initial_pose.clone()
        best_pose = current_pose.clone()
        poses = [current_pose.clone()]
        residuals = []
        damping = self.damping_init
        consecutive_failures = 0
        max_consecutive_failures = 5

        # 初始渲染 + 残差
        rendered_all = self._render_features_and_depth(current_pose)
        depth = rendered_all.pop('depth', depth_for_jac)
        if not self.use_rendered_depth:
            depth = depth_for_jac

        _, _, prev_residual, _ = self._compute_residual_and_normal_eq(
            query_feats, rendered_all, depth
        )
        best_residual = prev_residual
        residuals.append(prev_residual)

        for k in range(self.max_iters):
            # 1. 计算正规方程 (用当前渲染结果)
            JtJ, JtR, _, _ = self._compute_residual_and_normal_eq(
                query_feats, rendered_all, depth
            )

            # 2. LM 求解
            damping_mat = damping * torch.eye(6, device=device).unsqueeze(0)
            delta_xi = torch.linalg.solve(JtJ + damping_mat, JtR).squeeze(-1)
            step_size = delta_xi.norm().item()

            # 3. 试探性更新
            delta_T = se3_exp(delta_xi)
            candidate_pose = delta_T @ current_pose

            # 4. 渲染候选位姿 + 计算新残差
            candidate_rendered = self._render_features_and_depth(candidate_pose)
            candidate_depth = candidate_rendered.pop('depth', depth_for_jac)
            if not self.use_rendered_depth:
                candidate_depth = depth_for_jac

            _, _, new_residual, _ = self._compute_residual_and_normal_eq(
                query_feats, candidate_rendered, candidate_depth
            )

            # 5. LM 步接受/拒绝
            if new_residual < prev_residual:
                # 接受步 — 残差下降
                current_pose = candidate_pose
                rendered_all = candidate_rendered
                depth = candidate_depth
                prev_residual = new_residual
                damping = max(damping * self.damping_decrease, self.damping_min)
                consecutive_failures = 0
                accepted = True

                if new_residual < best_residual:
                    best_residual = new_residual
                    best_pose = current_pose.clone()
            else:
                # 拒绝步 — 增大阻尼，保持当前位姿
                damping = min(damping * self.damping_increase, self.damping_max)
                consecutive_failures += 1
                accepted = False

            residuals.append(new_residual if accepted else prev_residual)
            if accepted:
                poses.append(current_pose.clone())

            if verbose:
                status = "✓" if accepted else "✗"
                print(f"  [LM iter {k+1}] {status} residual={new_residual:.6f} "
                      f"step={step_size:.6f} rad ({step_size*180/3.14159:.3f}°) "
                      f"damping={damping:.1e}")

            # 6. 收敛/终止检查
            if accepted and step_size < self.convergence_thresh:
                if verbose:
                    print(f"  Converged at iter {k+1} (step < {self.convergence_thresh})")
                break

            if consecutive_failures >= max_consecutive_failures:
                if verbose:
                    print(f"  Stopped at iter {k+1} ({max_consecutive_failures} consecutive rejections)")
                break

        return {
            'final_pose': current_pose,
            'best_pose': best_pose,
            'poses': poses,
            'residuals': residuals,
            'num_iters': len(residuals) - 1,  # 减去初始残差
            'converged': (step_size < self.convergence_thresh) if residuals else False,
            'best_residual': best_residual,
        }

    @torch.no_grad()
    def align_fast(
        self,
        query_feats: Dict[str, torch.Tensor],
        initial_pose: torch.Tensor,
        depth_for_jac: torch.Tensor,
        verbose: bool = False,
    ) -> Dict[str, object]:
        """
        快速 Gauss-Newton 对准（不做 LM 候选检查，速度约 2× 快）

        每步：渲染 → 正规方程 → 求解 → 更新（无条件接受）
        跟踪残差最低的位姿作为 best_pose

        性能优化 v2:
        - 预计算 Image Jacobian（GT depth 时不变）
        - 预 normalize 查询特征（每帧只做一次）
        - 避免 .item() 的 CUDA sync，全用 tensor 比较
        - Cholesky 求解 6×6 系统（比 linalg.solve 快）
        - 内联 normal eq 避免函数调用开销

        Args:
            query_feats: {scale_name: (B, D, H, W)}
            initial_pose: (B, 4, 4)
            depth_for_jac: (B, H, W)
            verbose: bool

        Returns:
            dict with 'final_pose', 'best_pose', 'residuals', etc.
        """
        from pose_refine.utils.lie_algebra import se3_exp

        B = initial_pose.shape[0]
        device = initial_pose.device
        current_pose = initial_pose.clone()
        best_pose = current_pose.clone()
        best_residual_t = torch.tensor(float('inf'), device=device)
        residuals = []
        damping = self.damping_init

        # === 预计算不变量 ===
        # 1. Image Jacobian (GT depth 不变, 只算一次)
        Ju_cached, Jv_cached, valid_cached = compute_image_jacobian(
            depth_for_jac, self.intrinsics
        )
        N = Ju_cached.shape[1]

        # 2. 预 normalize 查询特征 (不变, 只做一次)
        query_normed = {}
        for scale in self.scale_names:
            if scale in query_feats:
                query_normed[scale] = F.normalize(query_feats[scale], p=2, dim=1)

        # 3. 预计算 damping identity
        eye6 = torch.eye(6, device=device).unsqueeze(0)  # (1, 6, 6)

        # 4. 预计算 ones for JtR (避免每次分配)
        valid_f = valid_cached.float()  # (B, N)

        prev_residual_t = torch.tensor(float('inf'), device=device)
        rel_patience_counter = 0

        for k in range(self.max_iters):
            # 1. 渲染
            rendered_all = self._render_features_and_depth(current_pose)
            rendered_all.pop('depth', None)  # 不用渲染深度

            # 使用 GT depth 和缓存的 Jacobian
            Ju, Jv, valid_f_iter = Ju_cached, Jv_cached, valid_f

            # 2. 内联正规方程 (避免函数调用 + 避免 .item() sync)
            total_JtJ = torch.zeros(B, 6, 6, device=device)
            total_JtR = torch.zeros(B, 6, 1, device=device)
            total_res_sq = torch.zeros(1, device=device)

            for scale in self.scale_names:
                if scale not in rendered_all or scale not in query_normed:
                    continue

                q = query_normed[scale]
                r_feat = rendered_all[scale]
                D = q.shape[1]

                residual = q - r_feat  # (B, D, H, W)
                grad_u, grad_v = compute_spatial_gradient(r_feat)

                gu = grad_u.reshape(B, D, N)
                gv = grad_v.reshape(B, D, N)
                res = residual.reshape(B, D, N)

                # 标量中间量 (对特征维度求和)
                A = (gu * gu).sum(1) * valid_f_iter   # (B, N)
                B_ = (gv * gv).sum(1) * valid_f_iter
                C = (gu * gv).sum(1) * valid_f_iter
                Ru = (gu * res).sum(1) * valid_f_iter
                Rv = (gv * res).sum(1) * valid_f_iter

                # JtJ: 4 个 bmm
                total_JtJ += (
                    torch.bmm((Ju * A.unsqueeze(-1)).transpose(1, 2), Ju) +
                    torch.bmm((Ju * C.unsqueeze(-1)).transpose(1, 2), Jv) +
                    torch.bmm((Jv * C.unsqueeze(-1)).transpose(1, 2), Ju) +
                    torch.bmm((Jv * B_.unsqueeze(-1)).transpose(1, 2), Jv)
                )

                # JtR
                total_JtR -= (
                    torch.bmm(Ju.transpose(1, 2), Ru.unsqueeze(-1)) +
                    torch.bmm(Jv.transpose(1, 2), Rv.unsqueeze(-1))
                )

                # 残差累加 (不调用 .item()!)
                total_res_sq += (res ** 2).sum()

            # 残差 (tensor, 不 sync)
            avg_residual_t = total_res_sq / max(N * len(self.scale_names), 1)

            # 跟踪最佳 (tensor 比较, 不 sync)
            if avg_residual_t < best_residual_t:
                best_residual_t = avg_residual_t.clone()
                best_pose = current_pose.clone()

            # 3. Cholesky 求解 (比 linalg.solve 快, JtJ 是正定的)
            A_mat = total_JtJ + damping * eye6
            try:
                L = torch.linalg.cholesky(A_mat)
                delta_xi = torch.cholesky_solve(total_JtR, L).squeeze(-1)  # (B, 6)
            except RuntimeError:
                # Cholesky 失败 (矩阵不正定), 回退到 solve
                delta_xi = torch.linalg.solve(A_mat, total_JtR).squeeze(-1)

            step_size_t = delta_xi.norm()

            if verbose:
                # 只在 verbose 时做 .item() sync
                print(f"  [GN iter {k+1}] residual={avg_residual_t.item():.6f} "
                      f"step={step_size_t.item():.6f} rad "
                      f"({step_size_t.item()*180/3.14159:.3f}°) "
                      f"damping={damping:.1e}")

            # 记录用于返回 (延迟 sync)
            residuals.append(avg_residual_t.detach())

            # 4. 无条件更新
            delta_T = se3_exp(delta_xi)
            current_pose = delta_T @ current_pose

            # 5. 自适应阻尼 (tensor 比较, 不 sync)
            if avg_residual_t > prev_residual_t:
                damping = min(damping * 3.0, self.damping_max)
            else:
                damping = max(damping / 3.0, self.damping_min)

            # 5b. 相对收敛检查 (tensor ops, 不 sync)
            rel_improvement = (prev_residual_t - avg_residual_t) / (prev_residual_t + 1e-10)
            if rel_improvement < self.rel_convergence_thresh:
                rel_patience_counter += 1
            else:
                rel_patience_counter = 0

            prev_residual_t = avg_residual_t.detach()

            # 6. 收敛检查 (step_size 或 relative convergence)
            if step_size_t.item() < self.convergence_thresh:
                if verbose:
                    print(f"  Converged (step) at iter {k+1}")
                break
            if rel_patience_counter >= self.rel_convergence_patience and k >= 5:
                # 至少运行 6 次迭代后才检查相对收敛
                if verbose:
                    print(f"  Converged (rel) at iter {k+1}, "
                          f"patience={rel_patience_counter}")
                break

        # 最终 sync: 转换 residuals 为 float list
        residuals_float = [r.item() for r in residuals]

        return {
            'final_pose': current_pose,
            'best_pose': best_pose,
            'residuals': residuals_float,
            'num_iters': len(residuals_float),
            'best_residual': best_residual_t.item(),
            'converged': step_size_t.item() < self.convergence_thresh if residuals else False,
        }

    def _render_features_and_depth(self, pose_w2c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """渲染多尺度特征 + 深度图"""
        B = pose_w2c.shape[0]
        result = {}
        for i, name in enumerate(self.scale_names):
            feat_list = []
            depth_list = []
            for b in range(B):
                # 第一个尺度同时渲染深度
                need_depth = (i == 0 and self.use_rendered_depth)
                r = self.renderer.render_scale(name, pose_w2c[b], return_depth=need_depth)
                feat_list.append(r['feature_map'])
                if need_depth and 'depth_map' in r:
                    depth_list.append(r['depth_map'])
            result[name] = torch.stack(feat_list, dim=0)
            if depth_list:
                # depth_map 分辨率可能不同于特征，需要 resize
                depth_raw = torch.stack(depth_list, dim=0)  # (B, dH, dW)
                _, H, W = result[name].shape[1], result[name].shape[2], result[name].shape[3]
                if depth_raw.shape[-2:] != (H, W):
                    depth_raw = F.interpolate(
                        depth_raw.unsqueeze(1), size=(H, W), mode='nearest'
                    ).squeeze(1)
                result['depth'] = depth_raw
        return result
