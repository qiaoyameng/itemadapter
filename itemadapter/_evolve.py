"""Structural "evolve" support for items.

Unlike ``ItemAdapter.__setitem__``, which mutates the wrapped object in place,
:func:`evolve` returns a *new* item with the requested field paths updated.
Branches that are not touched keep object identity, and each changed object is
rebuilt at most once. Paths may traverse any combination of mappings, lists,
``dataclass`` objects and ``attrs`` objects.

The work happens in two phases:

1. **Path location.** Every path is resolved against the original item. This
   phase only reads data; user constructors are never called, so nothing can
   be mutated even if validation fails later.
2. **Rebuild.** The located changes are applied bottom-up. Changed objects are
   rebuilt through their public constructor (so field declarations,
   ``init=False`` fields, default factories and frozen-type validation are all
   honored), and untouched branches keep the original object by identity.
"""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from itemadapter._imports import attr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from itemadapter.adapter import ItemAdapter as _ItemAdapter

__all__ = [
    "DELETE",
    "evolve",
    "EvolveError",
    "EvolveKeyError",
    "EvolvePathError",
    "EvolveTypeError",
]

_TOKEN_RE = re.compile(r"(?:(?P<dot>\.)|(?P<bracket>\[(?P<inner>[^\]]*)\]))")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INT_RE = re.compile(r"[+-]?[0-9]+")


class EvolveError(Exception):
    """Base class for errors raised while evolving an item."""


class EvolvePathError(EvolveError, ValueError):
    """Raised for malformed paths, or for conflicting parent/child updates."""


class EvolveTypeError(EvolveError, TypeError):
    """Raised when a path cannot be used with the object it reaches."""


class EvolveKeyError(EvolveError, KeyError):
    """Raised when a path points at a field or key that does not exist."""


class _DeleteField:
    """Sentinel requesting the removal of the reached field or key."""

    _instance = None

    def __new__(cls) -> "_DeleteField":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DELETE"

    def __bool__(self) -> bool:
        return False


DELETE = _DeleteField()


def _format_path(path: tuple) -> str:
    return repr(path)


def _parse_string_path(raw: str) -> tuple:
    """Parse a dotted/bracketed string into a path tuple.

    ``a.b`` -> ``("a", "b")``; ``a[0]["b"]`` -> ``("a", 0, "b")``.
    """
    if not raw:
        raise EvolvePathError("Path must not be empty")
    components: list = []
    pos = 0
    if raw[0] != "[":
        match = _NAME_RE.match(raw)
        if match is None:
            raise EvolvePathError(f"Invalid path: {raw!r}")
        components.append(match.group(0))
        pos = match.end()
    while pos < len(raw):
        token = _TOKEN_RE.match(raw, pos)
        if token is None or token.start() != pos:
            raise EvolvePathError(f"Invalid path: {raw!r}")
        pos = token.end()
        if token.group("dot") is not None:
            name = _NAME_RE.match(raw, pos)
            if name is None:
                raise EvolvePathError(f"Invalid path: {raw!r}")
            components.append(name.group(0))
            pos = name.end()
        else:
            inner = token.group("inner")
            if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in {"'", '"'}:
                try:
                    value = ast.literal_eval(inner)
                except (ValueError, SyntaxError) as ex:
                    raise EvolvePathError(f"Invalid path: {raw!r}") from ex
                if not isinstance(value, str):
                    raise EvolvePathError(f"Invalid path: {raw!r}")
            elif _INT_RE.fullmatch(inner):
                value = int(inner)
            else:
                raise EvolvePathError(f"Invalid path: {raw!r}")
            components.append(value)
    return tuple(components)


def _normalize_path(path: Any) -> tuple:
    if isinstance(path, str):
        return _parse_string_path(path)
    if isinstance(path, tuple):
        normalized = path
        for component in normalized:
            if isinstance(component, bool) or not isinstance(component, (str, int)):
                raise EvolvePathError(
                    "Path components must be strings or integers, "
                    f"got {type(component).__name__}: {component!r}"
                )
        return normalized
    raise EvolvePathError(
        f"Path must be a tuple or string, got {type(path).__name__}: {path!r}"
    )


def _build_trie(updates: "Mapping[Any, Any]") -> tuple[dict, Any, bool]:
    """Build a prefix trie out of the update mapping.

    Every node maps a path component to ``(value, children)``; an intermediate
    node has ``children`` as a dict, a terminal node has ``None``. Conflicts
    (a path that is both a terminal and a prefix of another path, including a
    duplicate) raise :class:`EvolvePathError` independently of the order in
    which updates were supplied.

    Return ``(trie, root_value, root_replaced)``.
    """
    trie: dict = {}
    root_value: Any = None
    root_replaced = False
    for raw_path, value in updates.items():
        path = _normalize_path(raw_path)
        if not path:
            if root_replaced:
                raise EvolvePathError("Duplicate root update path: ()")
            if trie:
                raise EvolvePathError("Root replacement conflicts with nested field updates")
            root_replaced = True
            root_value = value
            continue
        if root_replaced:
            raise EvolvePathError(
                f"Nested update {_format_path(path)} conflicts with root replacement"
            )
        node = trie
        for depth, component in enumerate(path):
            terminal = depth == len(path) - 1
            existing = node.get(component)
            prefix = path[: depth + 1]
            if terminal:
                if existing is not None:
                    raise EvolvePathError(
                        f"Duplicate update path: {_format_path(path)}"
                    )
                node[component] = (value, None)
            else:
                if existing is None:
                    child = {}
                    node[component] = (None, child)
                    node = child
                else:
                    if existing[1] is None:
                        raise EvolvePathError(
                            f"Path {_format_path(path)} conflicts with parent path "
                            f"{_format_path(prefix)}"
                        )
                    node = existing[1]
    return trie, root_value, root_replaced


@dataclass
class _NodePlan:
    """Pending changes for a single object, shared across all of its paths."""

    obj: Any
    kind: str
    sets: dict[Any, Any]
    deletes: set[Any]
    children: dict[Any, "_NodePlan"]


def _get_plan(obj: Any, registry: dict[int, _NodePlan]) -> _NodePlan:
    plan = registry.get(id(obj))
    if plan is None:
        plan = _NodePlan(obj=obj, kind="", sets={}, deletes=set(), children={})
        registry[id(obj)] = plan
    return plan


def _classify(obj: Any) -> str:
    if isinstance(obj, list):
        return "list"
    try:
        adapter = _item_adapter(obj)
    except TypeError:
        return "unsupported"
    from itemadapter.adapter import AttrsAdapter, DataclassAdapter, DictAdapter, PydanticAdapter

    if isinstance(adapter.adapter, DataclassAdapter):
        return "dataclass"
    if isinstance(adapter.adapter, AttrsAdapter):
        return "attrs"
    if isinstance(adapter.adapter, PydanticAdapter):
        return "pydantic"
    if isinstance(adapter.adapter, DictAdapter):
        return "mapping"
    return "unsupported"


def _adapter_contains(adapter: "_ItemAdapter", key: Any) -> bool:
    try:
        return key in adapter
    except TypeError:
        return False


def _item_adapter(obj: Any) -> "_ItemAdapter":
    # imported lazily to avoid an import cycle (adapter imports nothing here,
    # but the public package exports both)
    from itemadapter.adapter import ItemAdapter

    return ItemAdapter(obj)


def _field_in_init(obj: Any, kind: str, field_name: str) -> bool:
    if kind == "dataclass":
        import dataclasses

        field = next(
            (candidate for candidate in dataclasses.fields(obj) if candidate.name == field_name),
            None,
        )
        return field is not None and bool(field.init)
    attribute = _attrs_attribute(obj, field_name)
    return attribute is not None and bool(attribute.init)


def _attrs_attribute(obj: Any, field_name: str) -> Any:
    """Return the attrs attribute for a field name or public init alias."""
    fields_dict = attr.fields_dict(obj.__class__)
    if field_name in fields_dict:
        return fields_dict[field_name]
    for attribute in fields_dict.values():
        alias = getattr(attribute, "alias", None)
        if alias is not None and alias == field_name:
            return attribute
        if attribute.name.lstrip("_") == field_name:
            return attribute
    return None


def _attrs_init_name(attribute: Any) -> str:
    return getattr(attribute, "alias", None) or attribute.name.lstrip("_")


def _locate(
    obj: Any,
    kind: str,
    key: Any,
    path: tuple,
    *,
    delete: bool,
    traverse: bool,
) -> tuple[Any, str, Any]:
    """Validate a single path component during the location phase.

    Return ``(child, child_kind, normalized_key)``. Reads only; never writes
    and never invokes a user constructor. ``child`` is ``None`` when the
    component is a terminal set on a mapping key that does not exist yet.
    """
    if kind == "list":
        if not isinstance(key, int) or isinstance(key, bool):
            raise EvolvePathError(f"List indices must be integers: {_format_path(path)}")
        index = key + len(obj) if key < 0 else key
        upper = len(obj) - 1 if (traverse or delete) else len(obj)
        if index < 0 or index > upper:
            raise EvolveKeyError(
                f"List index out of range: {_format_path(path)} (list length {len(obj)})"
            )
        if index == len(obj):
            return None, "missing", index
        return obj[index], _classify(obj[index]), index

    if kind == "mapping":
        if not isinstance(key, (str, int)) or isinstance(key, bool):
            raise EvolvePathError(
                f"Mapping keys in paths must be strings or integers: {_format_path(path)}"
            )
        if key not in obj:
            if delete:
                raise EvolveKeyError(f"Key not found: {_format_path(path)}")
            if traverse:
                raise EvolveKeyError(f"Key not found: {_format_path(path)}")
            return None, "missing", key
        return obj[key], _classify(obj[key]), key

    if kind in {"dataclass", "attrs", "pydantic"}:
        if not isinstance(key, str):
            raise EvolvePathError(f"Field names must be strings: {_format_path(path)}")
        attribute_name = key
        if kind == "attrs":
            attribute = _attrs_attribute(obj, key)
            if attribute is None:
                raise EvolveKeyError(
                    f"{obj.__class__.__name__} does not support field: {key!r} "
                    f"({_format_path(path)})"
                )
            attribute_name = attribute.name
        elif not _adapter_contains(_item_adapter(obj), key):
            raise EvolveKeyError(
                f"{obj.__class__.__name__} does not support field: {key!r} "
                f"({_format_path(path)})"
            )
        if kind in {"dataclass", "attrs"} and not _field_in_init(obj, kind, attribute_name):
            raise EvolveTypeError(
                f"Cannot set, delete or traverse init=False field {attribute_name!r} on "
                f"{obj.__class__.__name__}: {_format_path(path)}"
            )
        if delete and not hasattr(obj, attribute_name):
            raise EvolveKeyError(
                f"{obj.__class__.__name__} field is not set: {attribute_name!r} "
                f"({_format_path(path)})"
            )
        if kind == "pydantic" and delete:
            raise EvolveTypeError(
                f"Cannot delete fields on {obj.__class__.__name__}: {_format_path(path)}"
            )
        child = getattr(obj, attribute_name)
        return child, _classify(child), attribute_name

    raise EvolveTypeError(f"Cannot traverse {type(obj).__name__}: {_format_path(path)}")


def _resolve(
    trie: dict,
    obj: Any,
    kind: str,
    path: tuple,
    registry: dict[int, _NodePlan],
) -> _NodePlan:
    """Locate every terminal path of ``trie`` under ``obj`` (read-only)."""
    plan = _get_plan(obj, registry)
    if not plan.kind:
        plan.kind = kind
    for key, (value, children) in trie.items():
        component_path = (*path, key)
        if children is not None:
            child, child_kind, normalized_key = _locate(
                obj, kind, key, component_path, delete=False, traverse=True
            )
            if child_kind == "unsupported":
                raise EvolveTypeError(
                    f"Cannot traverse {type(child).__name__}: {_format_path(component_path)}"
                )
            plan.children[normalized_key] = _resolve(
                children, child, child_kind, component_path, registry
            )
        elif value is DELETE:
            _, _, normalized_key = _locate(
                obj, kind, key, component_path, delete=True, traverse=False
            )
            plan.deletes.add(normalized_key)
        else:
            _, _, normalized_key = _locate(
                obj, kind, key, component_path, delete=False, traverse=False
            )
            plan.sets[normalized_key] = value
    return plan


# --- Rebuild phase -----------------------------------------------------------

_REBUILT_NONE = object()


def _rebuild(plan: _NodePlan, kind: str, rebuilt: dict[int, Any]) -> Any:
    """Rebuild ``plan.obj`` with its located changes, bottom-up.

    Objects without any effective change are returned unchanged. Rebuilt
    objects are created through their public constructor, so validation,
    default factories and frozen restrictions are never bypassed.
    """
    cache_key = id(plan.obj)
    if cache_key in rebuilt:
        cached = rebuilt[cache_key]
        return plan.obj if cached is _REBUILT_NONE else cached
    effective_sets = _effective_sets(plan, kind)
    new_children = {
        key: _rebuild(child_plan, child_plan.kind, rebuilt)
        for key, child_plan in plan.children.items()
    }
    has_changes = bool(effective_sets or plan.deletes) or any(
        new_children[key] is not plan.children[key].obj for key in plan.children
    )
    if not has_changes:
        rebuilt[cache_key] = _REBUILT_NONE
        return plan.obj

    if kind == "list":
        result = _rebuild_list(plan, effective_sets, new_children)
    elif kind == "mapping":
        result = _rebuild_mapping(plan, effective_sets, new_children)
    elif kind == "dataclass":
        result = _rebuild_dataclass(plan, effective_sets, new_children)
    elif kind == "attrs":
        result = _rebuild_attrs(plan, effective_sets, new_children)
    elif kind == "pydantic":
        result = _rebuild_pydantic(plan, effective_sets, new_children)
    else:  # pragma: no cover - rejected during resolution
        raise EvolveTypeError(f"Cannot rebuild object of type {type(plan.obj).__name__}")
    rebuilt[cache_key] = result
    return result


def _current_value(obj: Any, kind: str, key: Any) -> Any:
    if kind in {"list", "mapping"}:
        return obj[key]
    return getattr(obj, key)


def _effective_sets(plan: _NodePlan, kind: str) -> dict[Any, Any]:
    """Leaf sets whose value differs from the current value."""
    effective = {}
    for key, value in plan.sets.items():
        try:
            current = _current_value(plan.obj, kind, key)
        except (KeyError, AttributeError, IndexError):
            effective[key] = value
        else:
            if current is not value and current != value:
                effective[key] = value
    return effective


def _rebuild_list(
    plan: _NodePlan, effective_sets: dict[int, Any], new_children: dict[int, Any]
) -> Any:
    new_values: dict[int, Any] = {**effective_sets, **new_children}
    # Start from the untouched elements, drop deleted indices, substitute the
    # changed ones.
    result_values = []
    for index, value in enumerate(plan.obj):
        if index in plan.deletes:
            continue
        result_values.append(new_values.get(index, value))
    # Sets targeting the index right after the last element append to it.
    end = len(plan.obj)
    for index in sorted(index for index in new_values if index >= end):
        result_values.append(new_values[index])
    if type(plan.obj) is list:
        return result_values
    # Preserve list subclasses without invoking their possibly side-effecting
    # constructor: copy the instance (frozen/custom copy) then replace content
    # through the base list methods.
    new_obj = copy.copy(plan.obj)
    list.clear(new_obj)
    list.extend(new_obj, result_values)
    return new_obj


def _rebuild_mapping(
    plan: _NodePlan, effective_sets: dict[Any, Any], new_children: dict[Any, Any]
) -> Any:
    new_obj = copy.copy(plan.obj)
    for key in plan.deletes:
        type(plan.obj).__delitem__(new_obj, key)
    for key, value in effective_sets.items():
        type(plan.obj).__setitem__(new_obj, key, value)
    for key, value in new_children.items():
        type(plan.obj).__setitem__(new_obj, key, value)
    return new_obj


def _rebuild_dataclass(
    plan: _NodePlan, effective_sets: dict[str, Any], new_children: dict[str, Any]
) -> Any:
    import dataclasses

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(plan.obj):
        if not field.init:
            continue
        if field.name in plan.deletes:
            continue
        if field.name in effective_sets:
            kwargs[field.name] = effective_sets[field.name]
        elif field.name in new_children:
            kwargs[field.name] = new_children[field.name]
        else:
            kwargs[field.name] = getattr(plan.obj, field.name)
    return plan.obj.__class__(**kwargs)


def _rebuild_attrs(
    plan: _NodePlan, effective_sets: dict[Any, Any], new_children: dict[str, Any]
) -> Any:
    kwargs: dict[str, Any] = {}
    for attribute in attr.fields(plan.obj.__class__):
        if not attribute.init:
            continue
        if attribute.name in plan.deletes:
            continue
        init_name = _attrs_init_name(attribute)
        if attribute.name in effective_sets:
            kwargs[init_name] = effective_sets[attribute.name]
        elif attribute.name in new_children:
            kwargs[init_name] = new_children[attribute.name]
        else:
            kwargs[init_name] = getattr(plan.obj, attribute.name)
    return plan.obj.__class__(**kwargs)


def _rebuild_pydantic(
    plan: _NodePlan, effective_sets: dict[Any, Any], new_children: dict[str, Any]
) -> Any:
    updates: dict[str, Any] = dict(effective_sets)
    updates.update(new_children)
    if not updates:
        return plan.obj
    if hasattr(plan.obj, "model_copy"):
        return plan.obj.model_copy(update=updates)
    return plan.obj.copy(update=updates)


def evolve(item: Any, updates: "Mapping[Any, Any]") -> Any:
    """Return a copy of ``item`` with the given field paths updated.

    ``updates`` maps field paths to new values. Paths are tuples or
    dotted/bracketed strings of field names (``str``) and list indices
    (``int``), e.g. ``("meta", "tags", 0)`` or ``"meta.tags[0]"``. A path may
    freely mix mappings, lists, ``dataclass`` objects and ``attrs`` objects.

    Use :data:`DELETE` as a value to remove a field or key; an explicit
    ``None`` sets the field to ``None``. Missing fields/keys raise
    :class:`EvolveKeyError`. Untouched branches keep object identity, and each
    changed object is rebuilt at most once.

    All paths are located before anything is rebuilt, so parent/child path
    conflicts and validation failures leave the original item untouched.
    """
    if not hasattr(updates, "items"):
        raise EvolveTypeError(
            f"updates must be a mapping of paths to values, got {type(updates).__name__}"
        )

    trie, root_value, root_replaced = _build_trie(updates)
    if root_replaced:
        return root_value
    if not trie:
        return item

    kind = _classify(item)
    if kind == "unsupported":
        raise EvolveTypeError(f"Cannot evolve objects of type {type(item).__name__}")

    registry: dict[int, _NodePlan] = {}
    root_plan = _resolve(trie, item, kind, (), registry)
    return _rebuild(root_plan, kind, {})
