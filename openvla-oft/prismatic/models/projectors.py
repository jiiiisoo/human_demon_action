"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class PointcloudProjector(nn.Module):
    """
    Projects an input pointcloud token (flattened initial frame) into the LLM's embedding space.
    """

    def __init__(self, llm_dim: int, num_points: int, point_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.num_points = num_points
        self.point_dim = point_dim

        input_dim = self.num_points * self.point_dim
        self.fc1 = nn.Linear(input_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        # pointcloud: (bsz, num_points, point_dim)
        flattened = pointcloud.reshape(pointcloud.shape[0], -1)
        projected_features = self.fc1(flattened)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class PointNetProjector(nn.Module):
    """
    PointNet-style encoder that projects pointcloud into the LLM's embedding space.

    Unlike the simple PointcloudProjector which flattens the pointcloud,
    this uses the PointNet architecture:
    1. Per-point feature extraction via shared MLP
    2. Global max pooling to aggregate features
    3. Final projection to LLM embedding space

    This is permutation invariant and scales better with varying point counts.
    """

    def __init__(
        self,
        llm_dim: int,
        num_points: int,
        point_dim: int = 3,
        hidden_dims: tuple = (64, 128, 256),
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.num_points = num_points
        self.point_dim = point_dim

        # Per-point feature extraction (shared MLP)
        layers = []
        in_dim = point_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        self.point_mlp = nn.Sequential(*layers)

        # Final projection after global pooling
        self.global_fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pointcloud: (bsz, num_points, point_dim)
        Returns:
            (bsz, llm_dim)
        """
        bsz, n_pts, _ = pointcloud.shape

        # Per-point features: (bsz, num_points, point_dim) -> (bsz * num_points, point_dim)
        x = pointcloud.reshape(bsz * n_pts, -1)

        # Shared MLP with BatchNorm
        x = self.point_mlp(x)  # (bsz * num_points, hidden_dims[-1])

        # Reshape back: (bsz, num_points, hidden_dims[-1])
        x = x.reshape(bsz, n_pts, -1)

        # Global max pooling: (bsz, hidden_dims[-1])
        global_feat = x.max(dim=1)[0]

        # Project to LLM space
        output = self.global_fc(global_feat)
        return output


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
