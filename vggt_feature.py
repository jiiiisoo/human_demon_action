import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

# 1) load images (list of file paths)
# image_names = ["/workspace/LAPA/imgs/bottle_1.png", "/workspace/LAPA/imgs/biman_flower_1.png", "/workspace/LAPA/imgs/hand_1.png"]
image_names = ["/workspace/LAPA/imgs/bottle_1.png"]
imgs = load_and_preprocess_images(image_names).to(device)   # (T,3,H,W)
imgs = imgs[None]  # (B=1,S,3,H,W)

with torch.no_grad():
    # 2) get tokens per transformer layer and the patch start index
    aggregated_tokens_list, ps_idx = model.aggregator(imgs)   # ps_idx: patch_start_idx (int)

# 3) pick a layer to use as features
#    - last layer: global/high-level (what camera/depth heads use)
#    - earlier layer (e.g. -3): more local/detail
layer_idx = -1
# aggregated_tokens_list is a list of each layer's feature
# ps_idx is the index where patch tokens start (after camera + register tokens)
tokens = aggregated_tokens_list[layer_idx]  # (B,S,P,D) or (B,P,D)
patch_start = int(ps_idx.item()) if torch.is_tensor(ps_idx) else int(ps_idx)

# Slice out only patch tokens and keep per-frame tensors
print(tokens.shape)
if tokens.ndim == 4:
    B, S, P, D = tokens.shape
    patch_tokens = tokens[0, :, patch_start:, :]          # (S, P_patch, D)
    per_frame_feats = [patch_tokens[t] for t in range(S)]
else:
    B, P, D = tokens.shape
    patch_tokens = tokens[0, patch_start:, :].unsqueeze(0)  # (1, P_patch, D)
    per_frame_feats = [patch_tokens[0]]

# 5) (optional) unpatchify to H/16 x W/16 feature maps per frame
#    If your input images were HxW and VGGT uses 16x16 patches, grid size is (H//16, W//16).
#    ps_idx[t] tells you which of those grid locations were kept (if sub-sampled).
#    You can scatter back to a dense grid (missing patches remain zero).
# derive grid from token count instead of assuming H//patch * W//patch
patch = 16
H, W = imgs.shape[-2:]
Hp_nom = H // patch

per_frame_feat_maps = []
for feats_t in per_frame_feats:  # feats_t: (P_tokens, D)
    P_tokens, D = feats_t.shape
    # try using nominal Hp first
    if Hp_nom > 0 and P_tokens % Hp_nom == 0:
        Hp = Hp_nom
        Wp = P_tokens // Hp
        fmap_t = feats_t.view(Hp, Wp, D)
        per_frame_feat_maps.append(fmap_t)
    else:
        # fallback: try using nominal Wp
        Wp_nom = W // patch
        if Wp_nom > 0 and P_tokens % Wp_nom == 0:
            Wp = Wp_nom
            Hp = P_tokens // Wp
            fmap_t = feats_t.view(Hp, Wp, D)
            per_frame_feat_maps.append(fmap_t)
        else:
            # keep flat if we can't factor cleanly
            per_frame_feat_maps.append(feats_t)

# visualization (works for both grid and flat)
import cv2, numpy as np, torch

feat = per_frame_feat_maps[0]
if feat.ndim == 3:  # (Hp, Wp, D)
    heat = torch.linalg.vector_norm(feat, ord=2, dim=-1)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_u8 = (heat.cpu().numpy() * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    heat_up = cv2.resize(heat_color, (W, H), interpolation=cv2.INTER_NEAREST)
else:  # flat tokens -> make a 1-row strip or skip
    # make a 1×P strip for quick inspection
    m = torch.linalg.vector_norm(feat, ord=2, dim=-1)  # (P,)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    strip = (m.cpu().numpy() * 255).astype(np.uint8)[None, :]  # (1,P)
    heat_color = cv2.applyColorMap(strip, cv2.COLORMAP_VIRIDIS)
    heat_up = cv2.resize(heat_color, (W, H), interpolation=cv2.INTER_NEAREST)

img = (imgs[0,0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
overlay = cv2.addWeighted(img, 0.5, heat_up, 0.5, 0)
cv2.imwrite("/workspace/human_demon_action/feat_heat.png", heat_up)
cv2.imwrite("/workspace/human_demon_action/feat_overlay.png", overlay)
print("Saved feat_heat.png and feat_overlay.png")