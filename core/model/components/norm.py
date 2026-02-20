import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm (AdaLN): replaces RMSNorm's fixed scale/shift with
    action-conditioned ones. Applied at every transformer block so the action
    signal can't be washed out.

    Normal RMSNorm: weight * (x / rms(x))
    AdaLN:          (1 + gamma(cond)) * (x / rms(x)) + beta(cond)

    The 1+ and zero-init mean it starts as identity and learns deviations.
    """

    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.norm = RMSNorm(embed_dim, affine=False)
        # Projects conditioning vector to (gamma, beta), zero-init for stable start
        self.proj = nn.Linear(cond_dim, 2 * embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        """
        Args:
            x:    (B, T, N, embed_dim)
            cond: (B, T, cond_dim) — per-timestep conditioning (action embedding)
        Returns:
            (B, T, N, embed_dim)
        """
        # (B, T, cond_dim) -> (B, T, 2*embed_dim) -> unsqueeze -> (B, T, 1, 2*embed_dim)
        gamma, beta = self.proj(cond).unsqueeze(2).chunk(2, dim=-1)
        return (1 + gamma) * self.norm(x) + beta


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        """
        Root Mean Square Layer Normalization.

        Args:
            dim: Dimension of the input features
            eps: Small value to prevent division by zero
            affine: If True, includes learnable scale (weight). Set False when
                    used inside AdaLN, where gamma/beta replace the learned scale.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            torch.Tensor: Normalized tensor of shape (..., dim)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm if self.weight is not None else x_norm
