import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

from pydantic_core import to_jsonable_python

RegistryType = Literal["processor", "analyser", "checker", "communicator", "model"]

obj_type = type

REGISTRY_INFO = "__registry_info__"
REGISTRY_PARAMS = "__registry_params__"
_registry: dict[str, Callable[..., Any]] = {}


@dataclass
class RegistryInfo:
    type: RegistryType
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


def registry_key(type: RegistryType, name: str) -> str:
    return f"{type}:{name}"


def registry_get(registry_info: RegistryInfo) -> Callable[..., Any]:
    """
    Get a callable from the registry.

    Args:
        registry_info: The registry info.

    Returns:
        The registered callable (class or function).

    Raises:
        KeyError: If the requested object is not found in the registry.
    """
    key = registry_key(registry_info.type, registry_info.name)

    if key not in _registry:
        raise KeyError(
            f"No registered {registry_info.name} found with @{registry_info.type}"
        )

    return _registry[key]


def registry_add(o: Callable, info: RegistryInfo) -> None:
    r"""Add an object to the registry.

    Add the passed object to the registry using the RegistryInfo
    to index it for retrieval. The RegistryInfo is also added
    to the object as an attribute, which can retrieved by calling
    registry_info() on an object instance.

    Args:
        o (object): Object to be registered (Metric, Solver, etc.)
        info (RegistryInfo): Metadata (name, etc.) for object.
    """
    if not callable(o):
        raise TypeError(
            f"Object {o} is not callable. Only callables can be registered."
        )
    # tag the object
    setattr(o, REGISTRY_INFO, info)

    # add to registry
    _registry[registry_key(info.type, info.name)] = o


def registry_tag(
    type: Callable[..., Any],
    o: object,
    info: RegistryInfo,
    *args: Any,
    **kwargs: Any,
) -> None:
    r"""Tag an object w/ registry info.

    Tag the passed object with RegistryInfo. This function DOES NOT
    add the object to the registry (call registry_add() to both
    tag and add an object to the registry). Call registry_info()
    on a tagged/registered object to retrieve its info

    Args:
        type (T): type of object being tagged
        o (object): Object to be registered (Metric, Solver, etc.)
        info (RegistryInfo): Metadata (name, etc.) for object.
        *args (list[Any]): Creation arguments
        **kwargs (dict[str,Any]): Creation keyword arguments
    """
    # bind arguments to params
    named_params: dict[str, Any] = {}
    bound_params = inspect.signature(type).bind(*args, **kwargs)
    for param, value in bound_params.arguments.items():
        named_params[param] = registry_value(value)

    # callables are not serializable so use their names
    for param in named_params.keys():
        if is_registry_object(named_params[param]):
            named_params[param] = registry_info(named_params[param]).name
        elif callable(named_params[param]) and hasattr(named_params[param], "__name__"):
            named_params[param] = getattr(named_params[param], "__name__")
        elif isinstance(named_params[param], dict | list):
            named_params[param] = to_jsonable_python(
                named_params[param], fallback=lambda x: getattr(x, "__name__", None)
            )
        elif isinstance(named_params[param], str | int | float | str | bool | None):
            named_params[param] = named_params[param]
        else:
            named_params[param] = (
                getattr(named_params[param], "name", None)
                or getattr(named_params[param], "__name__", None)
                or getattr(obj_type(named_params[param]), "__name__", None)
                or "<unknown>"
            )

    # set attribute
    setattr(o, REGISTRY_INFO, info)
    setattr(o, REGISTRY_PARAMS, named_params)


def registry_value(o: object) -> Any:
    # treat tuple as list
    if isinstance(o, tuple):
        o = list(o)

    # recurse through collection types
    if isinstance(o, list):
        return [registry_value(x) for x in o]
    elif isinstance(o, dict):
        return {k: registry_value(v) for k, v in o.items()}
    else:
        return o


def is_registry_object(o: object, type: RegistryType | None = None) -> bool:
    r"""Check if an object is a registry object.

    Args:
        o (object): Object to lookup info for
        type: (RegistryType | None): Optional. Check for a specific type

    Returns:
        True if the object is a registry object (optionally of the specified
        type). Otherwise, False
    """
    info = getattr(o, REGISTRY_INFO, None)
    if info:
        reg_info = cast(RegistryInfo, info)
        if type:
            return reg_info.type == type
        else:
            return True
    else:
        return False


def registry_info(o: object) -> RegistryInfo:
    r"""Lookup RegistryInfo for an object.

    Args:
        o (object): Object to lookup info for

    Returns:
        RegistryInfo for object.
    """
    info = getattr(o, REGISTRY_INFO, None)
    if info is not None:
        return cast(RegistryInfo, info)
    else:
        name = getattr(o, "__name__", "unknown")
        raise ValueError(
            f"Object '{name}' does not have registry info. Did you forget to add a decorator somewhere?"
        )


def _import_path(path: str):
    """
    Import a file or package by the yser provided path exactly once and this will
    update the registry.

    This is for registering custom function.
    """
    p = Path(path).expanduser().resolve()
    key = str(p)
    if key in sys.modules:
        return sys.modules[key]

    if p.is_dir():
        # treat as a package directory
        spec = importlib.util.spec_from_file_location(p.name, p / "__init__.py")
    else:
        spec = importlib.util.spec_from_file_location(p.stem, p)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import custom module at {p}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
