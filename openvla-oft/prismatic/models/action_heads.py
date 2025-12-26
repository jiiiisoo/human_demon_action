"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred


class PointTrackingHead(nn.Module):
    """Predicts per-frame point tracking targets from action token hidden states."""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        num_points=64,
        tracking_dim=3,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_points = num_points
        self.tracking_dim = tracking_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=num_points * tracking_dim,
        )

    def predict_tracking(self, actions_hidden_states: torch.Tensor, pointcloud: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: Hidden states for action tokens, shape (B, chunk_len * action_dim, hidden_dim).
            pointcloud: Optional base pointcloud (unused for this head).

        Returns:
            torch.Tensor: Predicted tracking outputs with shape (B, chunk_len, num_points, tracking_dim).
        """
        batch_size = actions_hidden_states.shape[0]
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        tracking_flat = self.model(rearranged_actions_hidden_states)
        tracking = tracking_flat.reshape(batch_size, NUM_ACTIONS_CHUNK, self.num_points, self.tracking_dim)
        return tracking


class PointTrackingHeadWithPointInput(nn.Module):
    """Predicts per-frame point tracking targets conditioning on action states and base pointcloud."""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        point_hidden_dim=4096,
        num_points=64,
        tracking_dim=3,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_points = num_points
        self.tracking_dim = tracking_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(tracking_dim, point_hidden_dim),
            nn.ReLU(),
            nn.Linear(point_hidden_dim, hidden_dim),
        )
        self.ctx_mlp = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, tracking_dim),
        )

    def predict_tracking(self, actions_hidden_states: torch.Tensor, pointcloud: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            actions_hidden_states: Hidden states for action tokens, shape (B, chunk_len * action_dim, hidden_dim).
            pointcloud: Base pointcloud, shape (B, num_points, tracking_dim).

        Returns:
            torch.Tensor: Predicted tracking outputs with shape (B, chunk_len, num_points, tracking_dim).
        """
        if pointcloud is None:
            raise ValueError("PointTrackingHeadWithPointInput requires a pointcloud input.")
        batch_size = actions_hidden_states.shape[0]
        ctx = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        ctx_feat = self.ctx_mlp(ctx)  # (B, T, H)
        point_feat = self.point_mlp(pointcloud)  # (B, N, H)
        ctx_expanded = ctx_feat.unsqueeze(2).expand(-1, -1, self.num_points, -1)  # (B, T, N, H)
        pt_expanded = point_feat.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1, -1)  # (B, T, N, H)
        fusion = torch.cat([ctx_expanded, pt_expanded], dim=-1)  # (B, T, N, 2H)
        tracking = self.fusion_mlp(fusion)  # (B, T, N, tracking_dim)
        return tracking

class PointTrackingHeadParallel(nn.Module):
    """Parallel point tracking with transformer."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        num_points: int = 64,
        tracking_dim: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_points = num_points
        self.tracking_dim = tracking_dim
        self.hidden_dim = hidden_dim

        # Input embeddings
        self.point_embed = nn.Linear(tracking_dim, hidden_dim)
        self.action_embed = nn.Linear(input_dim * ACTION_DIM, hidden_dim)  # 7*D -> H
        
        # Positional embeddings
        self.point_idx_embed = nn.Parameter(torch.randn(1, num_points, hidden_dim) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, NUM_ACTIONS_CHUNK, hidden_dim) * 0.02)
        
        # Type embeddings
        self.type_embed = nn.ParameterDict({
            'p0': nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
            'action': nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
            'query': nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
        })

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, tracking_dim)

    def predict_tracking(
        self, 
        actions_hidden_states: torch.Tensor,  # (B, T * 7, D)
        pointcloud: torch.Tensor,              # (B, N, 3)
    ) -> torch.Tensor:
        if pointcloud is None:
            raise ValueError("PointTrackingHeadParallel requires a pointcloud input.")
            
        B = actions_hidden_states.shape[0]
        T = NUM_ACTIONS_CHUNK
        N = self.num_points

        # === Embed P0 ===
        p0 = self.point_embed(pointcloud) + self.point_idx_embed + self.type_embed['p0']
        # (B, N, H)

        # === Embed Actions ===
        actions = actions_hidden_states.reshape(B, T, -1)  # (B, T, 7*D)
        actions = self.action_embed(actions) + self.time_embed + self.type_embed['action']
        # (B, T, H)

        # === Create Queries ===
        queries = self.point_idx_embed.unsqueeze(1).expand(B, T, N, -1).clone()
        queries = queries + self.time_embed.unsqueeze(2)
        queries = queries + self.type_embed['query']
        queries = queries.reshape(B, T * N, -1)
        # (B, T*N, H)

        # === Build sequence: [P0] [Actions] [Queries] ===
        seq = torch.cat([p0, actions, queries], dim=1)
        # (B, N + T + T*N, H)

        # === Causal mask ===
        mask = self._build_mask(N, T, seq.device)

        # === Forward ===
        out = self.transformer(seq, mask=mask)

        # === Extract & reshape ===
        query_out = out[:, N + T:, :]
        query_out = query_out.reshape(B, T, N, -1)
        
        return self.output_proj(query_out)  # (B, T, N, 3)

    def _build_mask(self, N: int, T: int, device) -> torch.Tensor:
        total = N + T + T * N
        # True = 막힘 (attend 불가), False = 통과 (attend 가능)
        mask = torch.ones((total, total), dtype=torch.bool, device=device)
        
        # P0 ↔ P0
        mask[:N, :N] = False
        
        # Action_t sees: P0, Action_1..t
        for t in range(T):
            idx = N + t
            mask[idx, :N] = False
            mask[idx, N:N+t+1] = False
        
        # Query_t sees: P0, Action_1..t, Query_1..t
        for t in range(T):
            q_start = N + T + t * N
            q_end = q_start + N
            
            mask[q_start:q_end, :N] = False
            mask[q_start:q_end, N:N+t+1] = False
            mask[q_start:q_end, N+T:q_end] = False
        
        return mask

# class PointTrackingHead(nn.Module):
#     def __init__(
#         self,
#         input_dim=4096,
#         action_dim=7,
#         bottleneck_dim=1024,
#         num_points=5000,
#         tracking_dim=3,
#         num_blocks=3,
#     ):
#         super().__init__()
#         in_dim = input_dim * action_dim  # 4096 * 7 = 28672

#         self.down = nn.Linear(in_dim, bottleneck_dim)   # 28672 -> 1024


#         blocks = []
#         for _ in range(num_blocks):
#             blocks += [
#                 nn.LayerNorm(bottleneck_dim),
#                 nn.Linear(bottleneck_dim, bottleneck_dim),
#                 nn.ReLU(),
#             ]
#         self.mlp = nn.Sequential(*blocks)

#         self.out = nn.Linear(bottleneck_dim, num_points * tracking_dim)

#     def predict_tracking(self, actions_hidden_states):
#         B = actions_hidden_states.shape[0]
#         x = actions_hidden_states.reshape(B, NUM_ACTIONS_CHUNK, -1)   # (B, T, 28672)
#         x = self.down(x)                                              # (B, T, 1024)
#         x = self.mlp(x)                                               # (B, T, 1024)
#         tracking_flat = self.out(x)                                   # (B, T, 5000*3)
#         tracking = tracking_flat.view(B, NUM_ACTIONS_CHUNK, 5000, 3)
#         return tracking
