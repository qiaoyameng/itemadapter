from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from itemadapter._imports import attrs
from itemadapter._utils import _is_attrs_class

_UNSET: Any = object()


def _normalize_path(path: Iterable[str | int]) -> tuple[str | int, ...]:
    if isinstance(path, (str, bytes)) or not isinstance(path, Iterable):
        raise TypeError(
            f"path must be an iterable of strings and integers, got {path!r}"
        )
    steps = tuple(path)
    if not steps:
        raise ValueError("path must not be empty")
    for step in steps:
        if isinstance(step, bool) or not isinstance(step, (str, int)):
            raise TypeError(
                f"path steps must be strings or integers, got {step!r}"
            )
    return steps


def _raise_for_missing_field(obj: Any, field_name: str) -> None:
    cls_name = obj.__class__.__name__
    if hasattr(obj, field_name):
        raise TypeError(
            f"{cls_name}.{field_name} is a read-only attribute and cannot be "
            "modified with copy_with"
        )
    raise KeyError(f"{cls_name} does not support field: {field_name}")


def _copy_with_sequence(obj: list | tuple, head: Any, rest: tuple, value: Any) -> Any:
    if isinstance(head, bool) or not isinstance(head, int):
        raise TypeError(
            f"cannot use {head!r} as an index into a {obj.__class__.__name__}"
        )
    if not -len(obj) <= head < len(obj):
        raise IndexError(
            f"index {head} out of range for {obj.__class__.__name__} of length {len(obj)}"
        )
    if not rest:
        if value is _UNSET:
            raise TypeError(
                f"cannot delete elements from a {obj.__class__.__name__}"
            )
        new_element = value
    else:
        new_element = _copy_with(obj[head], rest, value)
    elements = list(obj)
    elements[head] = new_element
    return obj.__class__(elements)


def _rebuild_dict(obj: dict, new: dict) -> dict:
    cls = obj.__class__
    if cls is dict:
        return new
    try:
        return cls(new)
    except TypeError:
        return new


def _copy_with_dict(obj: dict, head: Any, rest: tuple, value: Any) -> dict:
    if not isinstance(head, str):
        raise TypeError(f"cannot use {head!r} as a key into a dict")
    new = dict(obj)
    if not rest:
        if value is _UNSET:
            if head not in obj:
                raise KeyError(head)
            del new[head]
        else:
            new[head] = value
        return _rebuild_dict(obj, new)
    if head not in obj:
        raise KeyError(head)
    new[head] = _copy_with(obj[head], rest, value)
    return _rebuild_dict(obj, new)


def _copy_with_dataclass(obj: Any, head: Any, rest: tuple, value: Any) -> Any:
    if not isinstance(head, str):
        raise TypeError(
            f"cannot use {head!r} as a field name of {obj.__class__.__name__}"
        )
    field_names = {field.name for field in dataclasses.fields(obj)}
    if head not in field_names:
        _raise_for_missing_field(obj, head)
    if not rest:
        if value is _UNSET:
            raise TypeError(
                f"cannot delete field {head!r} from {obj.__class__.__name__}"
            )
        change = value
    else:
        change = _copy_with(getattr(obj, head), rest, value)
    return dataclasses.replace(obj, **{head: change})


def _copy_with_attrs(obj: Any, head: Any, rest: tuple, value: Any) -> Any:
    if not isinstance(head, str):
        raise TypeError(
            f"cannot use {head!r} as a field name of {obj.__class__.__name__}"
        )
    if head not in attrs.fields_dict(obj.__class__):
        _raise_for_missing_field(obj, head)
    if not rest:
        if value is _UNSET:
            raise TypeError(
                f"cannot delete field {head!r} from {obj.__class__.__name__}"
            )
        change = value
    else:
        change = _copy_with(getattr(obj, head), rest, value)
    return attrs.evolve(obj, **{head: change})


def _copy_with(obj: Any, path: tuple, value: Any) -> Any:
    head, rest = path[0], path[1:]
    if isinstance(obj, (list, tuple)):
        return _copy_with_sequence(obj, head, rest, value)
    if isinstance(obj, dict):
        return _copy_with_dict(obj, head, rest, value)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _copy_with_dataclass(obj, head, rest, value)
    if attrs is not None and _is_attrs_class(obj.__class__):
        return _copy_with_attrs(obj, head, rest, value)
    raise TypeError(
        f"cannot traverse into object of type {obj.__class__.__name__}"
    )


def copy_with(item: Any, path: Iterable[str | int], value: Any = _UNSET) -> Any:
    """Return a copy of *item* with the value at *path* replaced by *value*.

    *path* is an iterable of string field names and integer indices that
    traverses supported items, dictionaries and sequences. The original item
    is never modified: only the branches along *path* are rebuilt, through the
    regular constructors, while untouched branches keep their identity.

    If *value* is not provided, the key at *path* is deleted instead, which is
    only supported for dictionary keys.
    """
    return _copy_with(item, _normalize_path(path), value)
