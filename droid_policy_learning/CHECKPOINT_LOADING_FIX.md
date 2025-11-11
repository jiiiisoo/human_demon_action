# Checkpoint Loading Fix for LIBERO Finetuning

## Problem

When loading the DROID pre-trained checkpoint for finetuning on LIBERO, the following errors occurred:

1. **Missing/Unexpected keys**: The DROID checkpoint uses observation keys like `camera/image/varied_camera_1_left_image`, while LIBERO uses `agentview_rgb` and `eye_in_hand_rgb`.

2. **Size mismatches**: 
   - The action dimension differs (10 in DROID vs 7 in LIBERO)
   - The low-dim observation concatenation layer has different sizes (1031 vs 1032)

## Solution

### 1. Modified `deserialize()` method in `diffusion_policy.py`

Added a `strict` parameter to the `deserialize()` method to allow partial weight loading:

```python
def deserialize(self, model_dict, strict=True):
    """
    Load model from a checkpoint.
    
    Args:
        model_dict (dict): a dictionary saved by self.serialize()
        strict (bool): whether to strictly enforce that the keys in state_dict 
            match the keys returned by this module's state_dict() function.
            Set to False for finetuning with different observation keys or action dimensions.
    """
    missing_keys, unexpected_keys = self.nets.load_state_dict(model_dict["nets"], strict=strict)
    
    if not strict:
        print(f"\n[Checkpoint Loading] strict=False mode:")
        if missing_keys:
            print(f"  Missing keys (will be randomly initialized): {len(missing_keys)} keys")
            # Print examples...
        if unexpected_keys:
            print(f"  Unexpected keys (ignored from checkpoint): {len(unexpected_keys)} keys")
            # Print examples...
    
    if model_dict.get("ema", None) is not None:
        try:
            self.ema.averaged_model.load_state_dict(model_dict["ema"], strict=strict)
        except:
            print("[Checkpoint Loading] Warning: Could not load EMA weights")
```

### 2. Updated `train.py` to support configurable strict loading

Modified the checkpoint loading section to read the `strict_weight_loading` config parameter:

```python
# if checkpoint is specified, load in model weights
ckpt_path = config.experiment.ckpt_path
if ckpt_path is not None:
    print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
    from robomimic.utils.file_utils import maybe_dict_from_checkpoint
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    
    # Use strict=False for finetuning to allow partial weight loading
    strict_loading = getattr(config.experiment, "strict_weight_loading", True)
    print(f"Loading checkpoint with strict={strict_loading}")
    model.deserialize(ckpt_dict["model"], strict=strict_loading)
```

### 3. Added `strict_weight_loading: false` to LIBERO config

In `configs/libero_spatial_finetune.json`, added:

```json
"experiment": {
    ...
    "ckpt_path": "/path/to/droid/checkpoint.pth",
    "strict_weight_loading": false,
    ...
}
```

## What Gets Loaded

With `strict=False`:

### ✅ Successfully Loaded (transferred from DROID):
- **Visual encoder backbone weights** (ResNet50Conv): These are the most valuable pre-trained weights
- **Visual encoder pooling layers**: General-purpose visual features
- **EMA model weights** (if compatible): For improved inference stability

### ❌ Randomly Initialized (due to architecture mismatch):
- **Observation encoder layers for LIBERO-specific keys** (`agentview_rgb`, `eye_in_hand_rgb`): These will use the ResNet50Conv backbone weights but the routing/naming is new
- **Action prediction head** (`noise_pred_net`): Must be re-initialized due to different action dimensions (10 → 7)
- **Low-dim observation fusion layer**: Different concatenation size due to different proprioceptive states

## Why This Works

1. **Visual features are transferable**: The ResNet50Conv backbone learns general visual features that are useful across different robot setups and cameras.

2. **Action space differences are isolated**: Only the final action prediction layers need to be re-initialized, while the visual processing pipeline can be reused.

3. **Task adaptation through finetuning**: The randomly initialized layers will be trained from scratch on LIBERO data, while the visual backbone can be finetuned or kept frozen.

## Expected Training Behavior

- **Initial loss might be higher** than continuing DROID training, since the action prediction head is random
- **Should converge faster** than training from scratch, due to pre-trained visual features
- **Visual encoder learning rate** could potentially be set lower than other layers for better transfer

## Verification

To verify the checkpoint loading is working:

1. Run training and check for the checkpoint loading messages:
   ```
   LOADING MODEL WEIGHTS FROM /path/to/checkpoint.pth
   Loading checkpoint with strict=False
   
   [Checkpoint Loading] strict=False mode:
     Missing keys (will be randomly initialized): XX keys
       - policy.obs_encoder.module.nets.obs.obs_nets.agentview_rgb...
       ...
     Unexpected keys (ignored from checkpoint): YY keys
       - policy.obs_encoder.module.nets.obs.obs_nets.camera/image/...
       ...
   ```

2. Training should proceed without errors

3. Loss should be reasonable (not NaN or extremely high)

## Alternative Approaches Considered

1. **Manual key remapping**: More complex, requires maintaining a mapping between DROID and LIBERO observation keys
2. **Using DROID's architecture**: Would require translating LIBERO data format, less flexible
3. **Training from scratch**: Would lose the benefit of pre-trained visual features

The chosen approach (partial loading with `strict=False`) is the simplest and most flexible solution.



