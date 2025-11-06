from functools import partial
import importlib
from typing import Any, Dict, Tuple, TypedDict, Union


class ModuleSpec(TypedDict):
    """
    Lightweight copy of Octo's ModuleSpec helper.

    module (str): module path of the callable.
    name (str): attribute name inside the module.
    args (tuple): positional args supplied at instantiation.
    kwargs (dict): keyword args supplied at instantiation.
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @staticmethod
    def create(
        callable_or_full_name: Union[str, callable], *args, **kwargs
    ) -> "ModuleSpec":  # type: ignore
        if isinstance(callable_or_full_name, str):
            if callable_or_full_name.count(":") != 1:
                raise ValueError(
                    "Expected fully-qualified import string like "
                    "'package.module:ClassName'."
                )
            module, name = callable_or_full_name.split(":")
        else:
            module, name = _infer_full_name(callable_or_full_name)

        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)

    @staticmethod
    def instantiate(spec: "ModuleSpec"):  # type: ignore
        if set(spec.keys()) != {"module", "name", "args", "kwargs"}:
            raise ValueError(
                f"Invalid ModuleSpec payload {spec}. "
                "Expected keys: module, name, args, kwargs."
            )
        cls = _import_from_string(spec["module"], spec["name"])
        return partial(cls, *spec["args"], **spec["kwargs"])

    @staticmethod
    def to_string(spec: "ModuleSpec"):  # type: ignore
        parts = []
        if spec["args"]:
            parts.append(", ".join(repr(a) for a in spec["args"]))
        if spec["kwargs"]:
            parts.append(", ".join(f"{k}={v!r}" for k, v in spec["kwargs"].items()))
        args_kwargs = ", ".join(parts)
        return f"{spec['module']}:{spec['name']}({args_kwargs})"


def _infer_full_name(obj: object):
    if hasattr(obj, "__module__") and hasattr(obj, "__name__"):
        return obj.__module__, obj.__name__
    raise ValueError(
        f"Could not infer import path for {obj}. "
        "Use a fully-qualified import string instead."
    )


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as exc:  # pragma: no cover - mirrors upstream helper
        raise ValueError(f"Could not import {module_string}:{name}") from exc
