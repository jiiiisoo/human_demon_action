import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from PIL import Image

# VGGT imports (assumes the vggt repo is available on PYTHONPATH)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _derive_grid(tokens_per_frame: int, H: int, W: int, patch: int = 16) -> Tuple[int, int]:
    """Derive (Hp, Wp) grid from token count and nominal patch size.

    Tries H//patch first; if not divisible, tries W//patch; otherwise
    returns a degenerate (tokens_per_frame, 1).
    """
    Hp_nom = max(H // patch, 1)
    if tokens_per_frame % Hp_nom == 0:
        Hp = Hp_nom
        Wp = tokens_per_frame // Hp
        return Hp, Wp
    Wp_nom = max(W // patch, 1)
    if tokens_per_frame % Wp_nom == 0:
        Wp = Wp_nom
        Hp = tokens_per_frame // Wp
        return Hp, Wp
    # fallback to a strip
    return tokens_per_frame, 1


class VGGTFeatureDataset(Dataset):
    """
    Dataset that returns VGGT feature pairs (curr,next) from frame folders.

    Each item is a tensor of shape [C_vggt, 2, Hp, Wp].

    Parameters
    ----------
    folder : str
        Root directory where each subfolder contains frames for one video.
    offset : int
        Temporal offset between the two frames to form a pair.
    cache_dir : Optional[str]
        If set, save precomputed features as .pt under this directory and load on reuse.
        Cache key is "{video_id}_{first_idx}_{second_idx}.pt".
    device : str
        Device to run VGGT on ("cuda" or "cpu").
    """

    def __init__(self, folder: str, offset: int = 5, cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.offset = offset
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy init VGGT to avoid paying cost in parent process when using workers.
        self._vggt = None

    def _ensure_model(self):
        if self._vggt is None:
            self._vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()

    def __len__(self):
        return len(self.folder_list)

    def _pick_pair(self, frames):
        first_idx = random.randint(0, len(frames) - self.offset - 1)
        first_idx = min(first_idx, len(frames) - self.offset - 1)
        second_idx = min(first_idx + self.offset, len(frames) - 1)
        return first_idx, second_idx

    def _cache_key(self, video_id: str, i: int, j: int) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{video_id}_{i}_{j}.pt"

    def _compute_vggt_pair(self, img_paths) -> torch.Tensor:
        """Compute VGGT feature pair for two image paths.

        Returns a tensor of shape [C_vggt, 2, Hp, Wp].
        """
        self._ensure_model()

        images = load_and_preprocess_images(list(img_paths)).to(self.device)  # [T=2,3,H,W]
        H, W = images.shape[-2:]

        with torch.no_grad():
            agg_list, patch_start = self._vggt.aggregator(images[None])  # [1,2,*,*], int
            tokens = agg_list[-1]  # choose last layer
            # tokens shape could be (B,S,P,D)
            if tokens.ndim == 4:
                _, S, P, D = tokens.shape
                patch_tokens = tokens[0, :, int(patch_start):, :]  # [S=2, P_patch, D]
            else:  # (B,P,D) corner case
                _, P, D = tokens.shape
                # duplicate single frame if needed (unlikely)
                pt = tokens[0, int(patch_start):, :]
                patch_tokens = torch.stack([pt, pt], dim=0)

            P_patch = patch_tokens.shape[1]
            Hp, Wp = _derive_grid(P_patch, H, W, patch=16)
            # reshape to (S, C_vggt, Hp, Wp)
            feat = patch_tokens.view(2, Hp, Wp, D).permute(0, 3, 1, 2).contiguous()
            # return as (C, 2, Hp, Wp)
            feat = feat.permute(1, 0, 2, 3).contiguous()
            return feat

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            video_id = self.folder_list[index]
            frame_dir = os.path.join(self.folder, video_id)
            frame_names = sorted(os.listdir(frame_dir))
            if len(frame_names) == 0:
                raise RuntimeError("empty folder")
            
            # pick current, next frame
            i, j = self._pick_pair(frame_names)
            p1 = os.path.join(frame_dir, frame_names[i])
            p2 = os.path.join(frame_dir, frame_names[j])

            if self.cache_dir is not None:
                key = self._cache_key(video_id, i, j)
                if key.exists():
                    return torch.load(key, map_location=self.device)

            feat = self._compute_vggt_pair((p1, p2))  # [C,2,Hp,Wp]
            if self.cache_dir is not None:
                torch.save(feat, key)
            return feat

        except Exception:
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


