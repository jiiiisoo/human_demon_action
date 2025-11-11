#!/usr/bin/env python
"""
Quick test to verify config loading works with strict_weight_loading parameter.
"""
import json
import sys
sys.path.insert(0, '/home/jisookim/human_demon_action/droid_policy_learning')

from robomimic.config import config_factory

# Load the LIBERO config
ext_cfg = json.load(open('/home/jisookim/human_demon_action/droid_policy_learning/configs/libero_spatial_finetune.json', 'r'))
config = config_factory(ext_cfg['algo_name'])

# Try to update with external config
with config.values_unlocked():
    config.update(ext_cfg)

print('✓ Config loaded successfully!')
print(f'✓ strict_weight_loading = {config.experiment.strict_weight_loading}')
print(f'✓ ckpt_path exists = {config.experiment.ckpt_path is not None}')
print(f'✓ skill_conditioning.enabled = {config.algo.skill_conditioning.enabled}')
print('\nConfig test passed! Ready to train.')



