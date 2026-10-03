from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from itemadapter.adapter import (
    AttrsAdapter,
    DataclassAdapter,
    DictAdapter,
    ItemAdapter,
    ScrapyItemAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from itemadapter.adapter import AdapterInterface

__all__ = ["DELETE", "evolve"]

_MISSING = object()


class _DeleteRequest:
    """Sentinel marking a path for deletion in an :func:`evolve` call."""

    _instance: "_DeleteRequest | None" = None

    def __new__(cls) -> "_DeleteRequest":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DELETE"

    def __bool__(self) -> bool:
        return False


DELETE = _DeleteRequest()


@dataclass
class _NodeState:
    """Information gathered, during the location phase, for a node that has
    to be rebuilt."""

    obj: Any
    kind: str
    depth: int
    adapter: "AdapterInterface | None" = None
    # key/index of a direct change at this node -> new value or DELETE
    changes: dict[Any, Any] = field(default_factory=dict)
    # insertion order of direct changes, for deterministic rebuilding
    change_order: list[Any] = field(default_factory=list)
    # key/index of a traversed direct child -> id() of the child node
    children: dict[Any, int] = field(default_factory=dict)


def evolve(item: Any, changes: "Mapping[Sequence, Any] | Sequence[tuple[Sequence, Any]]") -> Any:
    """Return a copy of *item* with *changes* applied.

    *changes* is either a mapping of paths to new values, or an iterable of
    ``(path, value)`` pairs. Each path is a non-empty sequence of dictionary
    keys, attribute names and list/tuple indices, e.g.
    ``("author", "addresses", 0, "city")``. Paths may freely mix dictionaries,
    dataclass/attrs items and lists or tuples.

    Nodes that are not touched by any change keep their identity, while every
    ancestor of a changed node is rebuilt exactly once, even when shared by
    several paths.

    Assigning the :data:`DELETE` sentinel to a path removes that key, index or
    field. Field deletion means the field is not passed to the constructor, so
    declared defaults and default factories apply, while fields without a
    default raise :class:`TypeError`. An explicit ``None`` value, in contrast,
    is set like any other value.

    Paths are located without constructing any user objects, and rebuilding
    goes through the real item constructors: field declarations, ``init=False``
    fields, default factories, ``__post_init__`` and frozen dataclass/attrs
    validation cannot be bypassed. If reconstruction fails, the original item
    and its nested mutable objects are left untouched.

    Parent/child path conflicts (one path replaces a node that another path
    descends into), as well as duplicate paths, raise :class:`ValueError`
    before anything is rebuilt.
    """
    normalized = _normalize_changes(changes)
    _check_conflicts(normalized)
    states: dict[int, _NodeState] = {}
    root_id = _locate(item, normalized, states)
    return _rebuild(item, root_id, states)


def _normalize_changes(
    changes: "Mapping[Sequence, Any] | Sequence[tuple[Sequence, Any]]]",
) -> list[tuple[tuple, Any]]:
    if hasattr(changes, "items"):
        pairs = list(changes.items())
    else:
        try:
            pairs = list(changes)
        except TypeError as ex:
            raise TypeError(
                "changes must be a mapping of paths to values or an iterable of "
                "(path, value) pairs"
            ) from ex
    normalized: list[tuple[tuple, Any]] = []
    for path, value in pairs:
        if isinstance(path, (str, bytes)) or not hasattr(path, "__iter__"):
            raise TypeError(
                "each path must be a non-empty sequence of keys, field names "
                f"and indices, got: {path!r}"
            )
        segments = tuple(path)
        if not segments:
            raise ValueError("paths must not be empty")
        for segment in segments:
            if isinstance(segment, bool) or not isinstance(segment, (str, int)):
                raise TypeError(
                    "path segments must be str keys/field names or int indices, "
                    f"got: {segment!r} in path {segments!r}"
                )
        normalized.append((segments, value))
    return normalized


def _check_conflicts(pairs: "list[tuple[tuple, Any]]") -> None:
    paths: set[tuple] = set()
    for path, _ in pairs:
        if path in paths:
            raise ValueError(f"duplicate path in changes: {list(path)!r}")
        paths.add(path)
    for path in paths:
        for length in range(1, len(path)):
            prefix = path[:length]
            if prefix in paths:
                raise ValueError(
                    f"conflicting paths in changes: {list(prefix)!r} replaces a node "
                    f"that {list(path)!r} descends into"
                )


def _classify(obj: Any, path: tuple) -> tuple[str, "AdapterInterface | None"]:
    if isinstance(obj, list):
        return "list", None
    if isinstance(obj, tuple):
        return "tuple", None
    try:
        wrapper = ItemAdapter(obj)
    except TypeError as ex:
        raise TypeError(
            f"cannot resolve path {list(path)!r}: value of type "
            f"{obj.__class__.__name__} does not support field access"
        ) from ex
    adapter = wrapper.adapter
    if isinstance(adapter, DataclassAdapter):
        return "dataclass", adapter
    if isinstance(adapter, AttrsAdapter):
        return "attrs", adapter
    if isinstance(adapter, (DictAdapter, ScrapyItemAdapter)):
        return "dictlike", adapter
    raise TypeError(
        f"cannot resolve path {list(path)!r}: items of type "
        f"{obj.__class__.__name__} are not supported by evolve()"
    )


def _get_state(
    obj: Any,
    path: tuple,
    depth: int,
    states: dict[int, _NodeState],
) -> _NodeState:
    obj_id = id(obj)
    state = states.get(obj_id)
    if state is None:
        kind, adapter = _classify(obj, path)
        state = _NodeState(obj=obj, kind=kind, depth=depth, adapter=adapter)
        states[obj_id] = state
    return state


def _resolve_index(index: int, length: int, path: tuple) -> int:
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise KeyError(index)
    return resolved


def _adapter_get(adapter: "AdapterInterface", key: Any, path: tuple) -> Any:
    try:
        return adapter[key]
    except AttributeError as ex:
        # declared dataclass/attrs field without a value set on the instance
        raise KeyError(key) from ex


def _locate(
    root: Any,
    pairs: "list[tuple[tuple, Any]]",
    states: dict[int, _NodeState],
) -> int:
    root_id = id(root)
    for path, value in pairs:
        node = root
        parent_state: _NodeState | None = None
        parent_key: Any = None
        current_path: tuple = ()
        for depth, segment in enumerate(path):
            current_path = path[: depth + 1]
            state = _get_state(node, current_path, depth, states)
            if parent_state is not None:
                parent_state.children.setdefault(parent_key, id(node))
            last = depth == len(path) - 1
            if last:
                if value is DELETE and not _contains(state, segment):
                    raise KeyError(segment)
                if segment not in state.change_order:
                    state.change_order.append(segment)
                state.changes[segment] = value
                break
            node = _descend(state, segment, current_path)
            parent_state = state
            parent_key = segment
    return root_id


def _contains(state: _NodeState, key: Any) -> bool:
    if state.kind in ("list", "tuple"):
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(
                f"indices into a {state.kind} must be int, got: {key!r}"
            )
        length = len(state.obj)
        index = key + length if key < 0 else key
        return 0 <= index < length
    if key in state.changes:
        return True
    adapter: "AdapterInterface" = state.adapter  # type: ignore[assignment]
    try:
        adapter[key]
    except (KeyError, AttributeError):
        return False
    return True


def _descend(state: _NodeState, key: Any, path: tuple) -> Any:
    if state.kind == "list" or state.kind == "tuple":
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(f"indices into a {state.kind} must be int, got: {key!r}")
        index = _resolve_index(key, len(state.obj), path)
        return state.obj[index]
    adapter: "AdapterInterface" = state.adapter  # type: ignore[assignment]
    return _adapter_get(adapter, key, path)


def _setattr_preserving_frozen(obj: Any, name: str, value: Any) -> None:
    # Same strategy as dataclasses.replace(): frozen items block regular
    # setattr with AttributeError, which object.__setattr__ bypasses while
    # construction-time validation has already run in __init__.
    try:
        setattr(obj, name, value)
    except AttributeError:
        # Frozen dataclass/attrs items block regular setattr; construction
        # (including __post_init__ and validators) already ran in __init__.
        object.__setattr__(obj, name, value)


def _resolve_child_value(
    state: _NodeState,
    key: Any,
    current: Any,
    rebuilt: dict[int, Any],
) -> Any:
    child_id = state.children.get(key)
    if child_id is not None:
        return rebuilt[child_id]
    return current


def _rebuild_dataclass(state: _NodeState, rebuilt: dict[int, Any]) -> Any:
    obj = state.obj
    cls = obj.__class__
    fields_dict = {dc_field.name: dc_field for dc_field in dataclasses.fields(cls)}
    init_kwargs: dict[str, Any] = {}
    non_init_values: dict[str, Any] = {}
    for name in state.change_order:
        if name not in fields_dict:
            raise KeyError(
                f"{cls.__name__} does not support field: {name}"
            )
    for name, dc_field in fields_dict.items():
        if state.changes.get(name) is DELETE:
            continue
        if name in state.changes:
            value = state.changes[name]
        else:
            current = getattr(obj, name, _MISSING)
            if current is _MISSING:
                # Let the constructor apply the default/factory, or complain.
                continue
            value = _resolve_child_value(state, name, current, rebuilt)
        if dc_field.init:
            init_kwargs[name] = value
        else:
            non_init_values[name] = value
    new = cls(**init_kwargs)
    for name, value in non_init_values.items():
        _setattr_preserving_frozen(new, name, value)
    return new


def _rebuild_attrs(state: _NodeState, rebuilt: dict[int, Any]) -> Any:
    from itemadapter._imports import attr

    obj = state.obj
    cls = obj.__class__
    attributes = {attribute.name: attribute for attribute in attr.fields(cls)}
    init_kwargs: dict[str, Any] = {}
    non_init_values: dict[str, Any] = {}
    for name in state.change_order:
        if name not in attributes:
            raise KeyError(f"{cls.__name__} does not support field: {name}")
    for name, attribute in attributes.items():
        if state.changes.get(name) is DELETE:
            continue
        if name in state.changes:
            value = state.changes[name]
        else:
            current = getattr(obj, name, _MISSING)
            if current is _MISSING:
                continue
            value = _resolve_child_value(state, name, current, rebuilt)
        if attribute.init:
            init_kwargs[attribute.alias] = value
        else:
            non_init_values[name] = value
    new = cls(**init_kwargs)
    for name, value in non_init_values.items():
        _setattr_preserving_frozen(new, name, value)
    return new


def _rebuild_dictlike(state: _NodeState, rebuilt: dict[int, Any]) -> Any:
    new = copy.copy(state.obj)
    adapter: "AdapterInterface" = type(state.adapter)(new)  # type: ignore[misc]
    for key, child_id in state.children.items():
        if key in state.changes:
            continue
        adapter[key] = rebuilt[child_id]
    for key in state.change_order:
        value = state.changes[key]
        if value is DELETE:
            del adapter[key]
        else:
            adapter[key] = value
    return new


def _rebuild_sequence(state: _NodeState, rebuilt: dict[int, Any]) -> Any:
    obj = state.obj
    length = len(obj)
    out: list[Any] = []
    changed_indices: dict[int, Any] = {}
    for key in state.change_order:
        index = _resolve_index(key, length, ())
        changed_indices[index] = state.changes[key]
    child_indices = {
        _resolve_index(key, length, ()): child_id
        for key, child_id in state.children.items()
    }
    for index, value in enumerate(obj):
        if index in changed_indices:
            change = changed_indices[index]
            if change is DELETE:
                continue
            out.append(change)
        elif index in child_indices:
            out.append(rebuilt[child_indices[index]])
        else:
            out.append(value)
    if state.kind == "list":
        new = copy.copy(obj)
        del new[:]
        new.extend(out)
        return new
    return type(obj)(out)


def _rebuild(root: Any, root_id: int, states: dict[int, _NodeState]) -> Any:
    if not states:
        return root
    ordered = sorted(states.values(), key=lambda state: -state.depth)
    rebuilt: dict[int, Any] = {}
    for state in ordered:
        if state.kind == "dataclass":
            new = _rebuild_dataclass(state, rebuilt)
        elif state.kind == "attrs":
            new = _rebuild_attrs(state, rebuilt)
        elif state.kind == "dictlike":
            new = _rebuild_dictlike(state, rebuilt)
        else:
            new = _rebuild_sequence(state, rebuilt)
        rebuilt[id(state.obj)] = new
    return rebuilt[root_id]
