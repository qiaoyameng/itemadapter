from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from itemadapter._imports import _scrapy_item_classes, attrs

if TYPE_CHECKING:
    from itemadapter.adapter import ItemAdapter

__all__ = ["MISSING", "copy_with"]


class _Missing:
    """Sentinel marking a field or key that must be removed."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()

PathComponent = str | int
_Change = tuple[tuple[PathComponent, ...], Any]


def _item_adapter() -> type[ItemAdapter]:
    # Imported lazily to avoid a circular import: itemadapter.adapter imports
    # this module to expose copy_with() on ItemAdapter.
    from itemadapter.adapter import ItemAdapter

    return ItemAdapter


def _is_model_item(obj: Any) -> bool:
    """Return True for supported items with a fixed set of fields.

    Plain mappings are handled by :class:`~itemadapter.adapter.DictAdapter`,
    but behave as containers for the purposes of copy_with(): their keys can
    be added and removed, unlike model fields. Scrapy items subclass ``dict``
    but have a fixed set of fields, so they count as model items.
    """
    if isinstance(obj, _scrapy_item_classes):
        return True
    return not isinstance(obj, Mapping) and _item_adapter().is_item(obj)


def copy_with(
    item: Any,
    *,
    path_changes: Mapping[Sequence[PathComponent], Any] | None = None,
    **field_changes: Any,
) -> Any:
    """Return a copy of ``item`` with the given changes applied.

    Never mutate ``item`` or any of its nested containers: every branch that
    holds a change is rebuilt through the item constructor, while unchanged
    branches keep their identity.
    """
    changes = _parse_changes(path_changes, field_changes)
    for path, value in changes:
        _validate_path(item, path, is_deletion=value is MISSING)
    root, changed = _apply_changes(item, changes, ())
    if not changed:
        return item
    return root


def _parse_changes(
    path_changes: Mapping[Sequence[PathComponent], Any] | None,
    field_changes: dict[str, Any],
) -> list[_Change]:
    changes: list[_Change] = []
    if path_changes is not None:
        for path, value in path_changes.items():
            changes.append((_normalize_path(path), value))
    for field_name, value in field_changes.items():
        changes.append(((field_name,), value))
    seen: set[tuple[PathComponent, ...]] = set()
    for path, _value in changes:
        if path in seen:
            raise ValueError(f"duplicate change for path {_format_path(path)}")
        seen.add(path)
    return changes


def _normalize_path(path: Any) -> tuple[PathComponent, ...]:
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise TypeError(
            "a path must be a sequence of str field names and int indices"
        )
    normalized = tuple(path)
    if not normalized:
        raise ValueError("a path must contain at least one component")
    for component in normalized:
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise TypeError(
                "path components must be str field names or int indices, got "
                f"{type(component).__name__}: {component!r}"
            )
    return normalized


def _format_path(path: Sequence[PathComponent]) -> str:
    return "[" + " -> ".join(repr(component) for component in path) + "]"


def _validate_path(
    item: Any, path: tuple[PathComponent, ...], *, is_deletion: bool
) -> None:
    """Walk the whole path before any rebuilding happens.

    Raise for unknown item fields, read-only computed attributes,
    non-traversable values and illegal deletions, so an invalid change fails
    without mutating or partially rebuilding the caller's objects.
    """
    current = item
    last_index = len(path) - 1
    for depth, component in enumerate(path):
        is_leaf = depth == last_index
        if isinstance(component, str):
            current = _validate_string_component(
                current, component, is_leaf=is_leaf, is_deletion=is_deletion
            )
        else:
            current = _validate_int_component(
                current, component, is_leaf=is_leaf, is_deletion=is_deletion
            )


def _validate_string_component(
    obj: Any, component: str, *, is_leaf: bool, is_deletion: bool
) -> Any:
    if _is_model_item(obj):
        adapter = _item_adapter()(obj)
        if component in adapter.field_names():
            if is_leaf and is_deletion:
                raise KeyError(
                    f"{obj.__class__.__name__}.{component} cannot be deleted, "
                    "fields of items are required"
                )
            return adapter[component]
        class_name = obj.__class__.__name__
        if hasattr(obj, component):
            raise KeyError(
                f"{class_name}.{component} is a read-only attribute, "
                "not a settable field"
            )
        raise KeyError(f"{class_name} does not support field: {component}")
    if isinstance(obj, Mapping):
        if component not in obj:
            if is_leaf and not is_deletion:
                # New mapping keys are allowed, so nothing more to check.
                return None
            raise KeyError(component)
        return obj[component]
    if isinstance(obj, (str, bytes)):
        raise TypeError(
            f"{type(obj).__name__} values cannot be traversed by field name"
        )
    raise TypeError(
        f"{type(obj).__name__} objects do not support field access: {component!r}"
    )


def _validate_int_component(
    obj: Any, component: int, *, is_leaf: bool, is_deletion: bool
) -> Any:
    if isinstance(obj, (set, frozenset)):
        raise TypeError(f"{type(obj).__name__} objects are not subscriptable")
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        length = len(obj)
        index = component + length if component < 0 else component
        if not 0 <= index < length:
            raise IndexError(
                f"index {component} out of range for {type(obj).__name__} "
                f"of length {length}"
            )
        return obj[component]
    if isinstance(obj, Mapping):
        if component not in obj:
            if is_leaf and not is_deletion:
                return None
            raise KeyError(component)
        return obj[component]
    raise TypeError(
        f"{type(obj).__name__} objects do not support integer indexing"
    )


def _apply_changes(
    obj: Any,
    changes: list[_Change],
    prefix: tuple[PathComponent, ...],
) -> tuple[Any, bool]:
    """Rebuild ``obj`` applying the changes whose paths start with ``prefix``.

    Return the rebuilt value and whether anything changed. Unchanged nested
    values are passed through untouched, which preserves object identity.
    """
    own_changes = [
        (path[len(prefix) :], value)
        for path, value in changes
        if path[: len(prefix)] == prefix
    ]
    if not own_changes:
        return obj, False

    leaf_changes: dict[PathComponent, Any] = {}
    nested_changes: dict[PathComponent, list[_Change]] = {}
    for relative_path, value in own_changes:
        component = relative_path[0]
        if len(relative_path) == 1:
            leaf_changes[component] = value
        else:
            nested_changes.setdefault(component, []).append((relative_path, value))

    if _is_model_item(obj):
        return _rebuild_item(obj, leaf_changes, nested_changes), True
    if isinstance(obj, Mapping):
        return _rebuild_mapping(obj, leaf_changes, nested_changes), True
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return _rebuild_sequence(obj, leaf_changes, nested_changes), True
    raise TypeError(
        f"{type(obj).__name__} objects cannot be rebuilt through copy_with()"
    )


def _rebuild_item(
    item: Any,
    leaf_changes: dict[PathComponent, Any],
    nested_changes: dict[PathComponent, list[_Change]],
) -> Any:
    adapter = _item_adapter()(item)
    replacements: dict[str, Any] = {}
    for component, value in leaf_changes.items():
        field_name = _require_item_field(adapter, item, component)
        if value is not MISSING:
            replacements[field_name] = value
    for component, component_changes in nested_changes.items():
        field_name = _require_item_field(adapter, item, component)
        rebuilt, _ = _apply_changes(
            adapter[field_name],
            [(relative_path[1:], value) for relative_path, value in component_changes],
            (),
        )
        replacements[field_name] = rebuilt
    return _construct(item, replacements)


def _require_item_field(
    adapter: ItemAdapter, item: Any, component: PathComponent
) -> str:
    if not isinstance(component, str):
        raise TypeError(
            f"{item.__class__.__name__} objects do not support integer indexing"
        )
    if component not in adapter.field_names():
        class_name = item.__class__.__name__
        if hasattr(item, component):
            raise KeyError(
                f"{class_name}.{component} is a read-only attribute, "
                "not a settable field"
            )
        raise KeyError(f"{class_name} does not support field: {component}")
    return component


def _rebuild_mapping(
    mapping: Mapping[Any, Any],
    leaf_changes: dict[PathComponent, Any],
    nested_changes: dict[PathComponent, list[_Change]],
) -> Mapping[Any, Any]:
    result = dict(mapping)
    for component, value in leaf_changes.items():
        if value is MISSING:
            del result[component]
        else:
            result[component] = value
    for component, component_changes in nested_changes.items():
        rebuilt, _ = _apply_changes(
            result[component],
            [(relative_path[1:], value) for relative_path, value in component_changes],
            (),
        )
        result[component] = rebuilt
    return type(mapping)(result)


def _rebuild_sequence(
    sequence: Sequence[Any],
    leaf_changes: dict[PathComponent, Any],
    nested_changes: dict[PathComponent, list[_Change]],
) -> Sequence[Any]:
    result = list(sequence)
    for component, value in leaf_changes.items():
        if value is MISSING:
            del result[component]
        else:
            result[component] = value
    for component, component_changes in nested_changes.items():
        rebuilt, _ = _apply_changes(
            result[component],
            [(relative_path[1:], value) for relative_path, value in component_changes],
            (),
        )
        result[component] = rebuilt
    return type(sequence)(result)


def _construct(item: Any, replacements: dict[str, Any]) -> Any:
    """Return a copy of ``item`` with ``replacements`` applied, going through
    the item constructor so that validation and defaults behave as usual."""
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return dataclasses.replace(item, **replacements)
    if attrs is not None and attrs.has(item.__class__):
        # evolve() passes changes to __init__(), which expects the init alias
        # of each attribute, not the attribute name (e.g. "x" for "_x").
        aliases = {field.name: field.alias for field in attrs.fields(item.__class__)}
        return attrs.evolve(
            item, **{aliases.get(name, name): value for name, value in replacements.items()}
        )
    if isinstance(item, dict):
        new_item = type(item)(item)
        new_item.update(replacements)
        return new_item
    return _construct_with_adapter(item, replacements)


def _construct_with_adapter(item: Any, replacements: dict[str, Any]) -> Any:
    """Fallback for items without a dedicated rebuild helper (e.g. Scrapy
    items, pydantic models): collect the current values and rebuild through
    the item constructor."""
    adapter = _item_adapter()(item)
    values = {
        field_name: adapter[field_name]
        for field_name in adapter.field_names()
        if field_name in adapter
    }
    values.update(replacements)
    return type(item)(**values)
