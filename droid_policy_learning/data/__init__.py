"""
Local copy of the RLDS data pipeline utilities used for training.
"""
from importlib import import_module

_LAZY_IMPORTS = {
    "apply_frame_transforms": "droid_policy_learning.data.datasets",
    "apply_trajectory_transforms": "droid_policy_learning.data.datasets",
    "make_dataset_from_rlds": "droid_policy_learning.data.datasets",
    "make_interleaved_dataset": "droid_policy_learning.data.datasets",
    "combine_dataset_statistics": "droid_policy_learning.data.data_utils",
    "NormalizationType": "droid_policy_learning.data.data_utils",
    "get_dataset_statistics": "droid_policy_learning.data.data_utils",
    "normalize_action_and_proprio": "droid_policy_learning.data.data_utils",
    "tree_map": "droid_policy_learning.data.data_utils",
    "tree_merge": "droid_policy_learning.data.data_utils",
    "ModuleSpec": "droid_policy_learning.data.spec",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    module = import_module(_LAZY_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
