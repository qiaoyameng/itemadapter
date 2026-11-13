from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, Final, cast

from itemadapter._imports import attrs, pydantic
from itemadapter.adapter import ItemAdapter

__all__ = [
    "GRAPH_FORMAT_VERSION",
    "GraphError",
    "GraphFormatError",
    "GraphRestorationError",
    "build_item_registry",
    "graph_from_item",
    "item_from_graph",
]


GRAPH_FORMAT_VERSION: Final = 1

_REF_KEY: Final = "$ref"
_TYPE_KEY: Final = "type"
_FIELDS_KEY: Final = "fields"
_VERSION_KEY: Final = "version"
_NODES_KEY: Final = "nodes"
_ROOT_KEY: Final = "root"
_MAX_DEPTH: Final = 1000


class GraphError(Exception):
    """Base class for errors raised by the graph export/restore format."""


class GraphFormatError(GraphError):
    """Raised when an object cannot be represented in the graph format."""


class GraphRestorationError(GraphError):
    """Raised when a graph document cannot be safely restored.

    The error is always raised *before* any item object is constructed, so a
    failed restoration never leaves partially built objects behind.
    """


def _is_frozen_class(item_class: type) -> bool:
    """Return True if instances of the class cannot be mutated after init."""
    if dataclasses.is_dataclass(item_class):
        return bool(getattr(getattr(item_class, "__dataclass_params__", None), "frozen", False))
    if attrs is not None and attrs.has(item_class):
        return getattr(item_class.__setattr__, "__name__", "") == "_frozen_setattrs"
    if pydantic is not None and issubclass(item_class, pydantic.BaseModel):
        return bool(item_class.model_config.get("frozen"))
    return False


def build_item_registry(
    item_classes: Iterable[type] | Mapping[str, type], *, frozen: Iterable[type] = ()
) -> dict[str, tuple[type[Any], bool]]:
    """Build a name -> (item class, is frozen) registry from item classes.

    The registry is the only source of model types accepted by
    :func:`item_from_graph`: type names found in a graph document are looked up
    here, never imported from module names stored in the payload.

    ``item_classes`` is an iterable of item classes, or a mapping of names to
    item classes. Class names are used as type names by default; two classes
    sharing a name raise :class:`GraphFormatError`, in which case a mapping
    with explicit names must be used instead.

    ``frozen`` is an iterable of classes that must be treated as frozen even
    when their frozenness cannot be introspected (e.g. custom adapter
    classes).
    """
    forced_frozen = set(frozen)
    if isinstance(item_classes, dict):
        entries = list(item_classes.items())
    else:
        entries = []
        for item_class in item_classes:
            if not isinstance(item_class, type):
                raise GraphFormatError(
                    f"Registry entry is not a class: {item_class!r}"
                )
            entries.append((item_class.__name__, item_class))

    registry: dict[str, tuple[type[Any], bool]] = {}
    for type_name, item_class in entries:
        if not isinstance(type_name, str) or not type_name:
            raise GraphFormatError(f"Invalid type name: {type_name!r}")
        if not isinstance(item_class, type):
            raise GraphFormatError(
                f"Registry entry for {type_name!r} is not a class: {item_class!r}"
            )
        if type_name in registry:
            raise GraphFormatError(f"Duplicate type name in registry: {type_name!r}")
        is_frozen = item_class in forced_frozen or _is_frozen_class(item_class)
        registry[type_name] = (item_class, is_frozen)
    return registry


def _encode_key(key: object) -> str:
    if not isinstance(key, str):
        raise GraphFormatError(
            f"Only string keys are supported in the JSON graph format, got {type(key).__name__}"
        )
    if key.startswith("$"):
        return f"${key}"
    return key


def _decode_key(key: str) -> str:
    if key.startswith("$$"):
        return key[1:]
    return key


def graph_from_item(root: object) -> dict[str, Any]:
    """Export an item and everything reachable from it as a graph document.

    Unlike :meth:`ItemAdapter.asdict`, shared item/dict objects are exported
    once and referenced elsewhere by their id, which preserves object identity
    and supports reference cycles. The returned document only contains JSON
    scalars, lists and dicts, and can be passed to :func:`json.dumps`.
    """
    if not ItemAdapter.is_item(root):
        raise GraphFormatError(f"The root must be a supported item, got {type(root).__name__}")

    nodes: list[dict[str, Any]] = []
    object_ids: dict[int, str] = {}
    type_names: dict[type[Any], str] = {}
    next_id = 0

    def register_type(item_class: type[Any]) -> str:
        type_name = type_names.get(item_class)
        if type_name is not None:
            return type_name
        type_name = item_class.__name__
        for other_class, other_name in type_names.items():
            if other_name == type_name:
                raise GraphFormatError(
                    "Two item classes share the same name, which cannot be "
                    f"unambiguously exported: {item_class!r} and {other_class!r}"
                )
        type_names[item_class] = type_name
        return type_name

    def encode(obj: object, depth: int) -> object:
        nonlocal next_id
        if depth > _MAX_DEPTH:
            raise GraphFormatError("Maximum nesting depth exceeded while exporting graph")
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, (list, dict)) or ItemAdapter.is_item(obj):
            obj_id = object_ids.get(id(obj))
            if obj_id is not None:
                return {_REF_KEY: obj_id}
            obj_id = str(next_id)
            next_id += 1
            object_ids[id(obj)] = obj_id
            if isinstance(obj, dict):
                fields = {
                    _encode_key(key): encode(value, depth + 1)
                    for key, value in obj.items()
                }
                nodes.append({"id": obj_id, _TYPE_KEY: "dict", _FIELDS_KEY: fields})
            elif isinstance(obj, list):
                nodes.append(
                    {
                        "id": obj_id,
                        _TYPE_KEY: "list",
                        _FIELDS_KEY: [encode(value, depth + 1) for value in obj],
                    }
                )
            else:
                adapter = ItemAdapter(obj)
                type_name = register_type(obj.__class__)
                fields = {
                    _encode_key(key): encode(value, depth + 1)
                    for key, value in adapter.items()
                }
                nodes.append({"id": obj_id, _TYPE_KEY: type_name, _FIELDS_KEY: fields})
            return {_REF_KEY: obj_id}
        raise GraphFormatError(
            f"Values of type {type(obj).__name__} are not supported by the JSON graph format"
        )

    root_ref = encode(root, 0)
    return {
        _VERSION_KEY: GRAPH_FORMAT_VERSION,
        _NODES_KEY: nodes,
        _ROOT_KEY: root_ref,
    }


def _resolve_ref(value: object, built: dict[str, object]) -> object:
    if isinstance(value, dict) and set(value.keys()) == {_REF_KEY}:
        return built[value[_REF_KEY]]
    if isinstance(value, list):
        return [_resolve_ref(item, built) for item in value]
    return value


def _create_shell(item_class: type[Any]) -> object:
    """Create an empty instance without running the constructor."""
    if dataclasses.is_dataclass(item_class):
        return object.__new__(item_class)
    if attrs is not None and attrs.has(item_class):
        return object.__new__(item_class)
    if pydantic is not None and issubclass(item_class, pydantic.BaseModel):
        return item_class.model_construct()
    if issubclass(item_class, dict):
        return item_class()
    return object.__new__(item_class)


def item_from_graph(
    document: object,
    registry: Mapping[str, tuple[type[Any], bool]],
    *,
    max_nodes: int | None = None,
) -> object:
    """Restore an item graph previously produced by :func:`graph_from_item`.

    Every structural and safety check (duplicate ids, dangling references,
    unknown types, node budget, un-restorable frozen cycles) runs before any
    item is constructed, so an invalid document raises
    :class:`GraphRestorationError` without returning a half-built object.

    ``registry`` maps the type names used in the document to an item class and
    its frozen flag, as built by :func:`build_item_registry`. Types are never
    imported from the payload itself.
    """
    if not isinstance(document, dict):
        raise GraphRestorationError("Graph document must be a dict")
    version = document.get(_VERSION_KEY)
    if version != GRAPH_FORMAT_VERSION:
        raise GraphRestorationError(
            f"Unsupported graph format version: {version!r}; expected {GRAPH_FORMAT_VERSION}"
        )

    raw_nodes = document.get(_NODES_KEY)
    root = document.get(_ROOT_KEY)
    if not isinstance(raw_nodes, list):
        raise GraphRestorationError("The 'nodes' entry must be a list")
    if max_nodes is not None and len(raw_nodes) > max_nodes:
        raise GraphRestorationError(
            f"Graph has {len(raw_nodes)} nodes, exceeding the max_nodes budget of {max_nodes}"
        )
    if not isinstance(root, dict) or set(root.keys()) != {_REF_KEY}:
        raise GraphRestorationError("The 'root' entry must be a reference object")

    parsed: dict[str, tuple[str, object]] = {}
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            raise GraphRestorationError(f"Node at position {index} must be a dict")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise GraphRestorationError(f"Node at position {index} has an invalid id")
        if node_id in parsed:
            raise GraphRestorationError(f"Duplicate node id: {node_id!r}")
        type_name = node.get(_TYPE_KEY)
        if not isinstance(type_name, str):
            raise GraphRestorationError(f"Node {node_id!r} has an invalid type")
        fields = node.get(_FIELDS_KEY)
        if type_name == "list":
            if not isinstance(fields, list):
                raise GraphRestorationError(f"Node {node_id!r} fields must be a list")
        elif type_name == "dict":
            if not isinstance(fields, dict):
                raise GraphRestorationError(f"Node {node_id!r} fields must be a dict")
        elif type_name not in registry:
            raise GraphRestorationError(f"Unknown item type: {type_name!r}")
        elif not isinstance(fields, dict):
            raise GraphRestorationError(f"Node {node_id!r} fields must be a dict")
        parsed[node_id] = (type_name, fields)

    root_id = root[_REF_KEY]
    if not isinstance(root_id, str) or root_id not in parsed:
        raise GraphRestorationError("The root reference does not point to an existing node")

    def collect_refs(value: object) -> set[str]:
        refs: set[str] = set()
        if isinstance(value, dict):
            keys = set(value.keys())
            if _REF_KEY in keys:
                if keys != {_REF_KEY}:
                    raise GraphRestorationError(
                        f"A {_REF_KEY!r} marker must be the only key of its object"
                    )
                target = value[_REF_KEY]
                if not isinstance(target, str) or target not in parsed:
                    raise GraphRestorationError(f"Dangling reference: {target!r}")
                refs.add(target)
            else:
                for nested in value.values():
                    refs.update(collect_refs(nested))
        elif isinstance(value, list):
            for item in value:
                refs.update(collect_refs(item))
        return refs

    dependencies: dict[str, set[str]] = {}
    for node_id, (_, fields) in parsed.items():
        dependencies[node_id] = collect_refs(fields)

    frozen_nodes = {
        node_id
        for node_id, (type_name, _) in parsed.items()
        if type_name in registry and registry[type_name][1]
    }

    # A cycle through a frozen node cannot be patched after init, so it has to
    # be rejected before any object is constructed.
    index_counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}

    def strong_connect(node_id: str) -> None:
        nonlocal index_counter
        indices[node_id] = index_counter
        low_links[node_id] = index_counter
        index_counter += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for target in dependencies[node_id]:
            if target not in indices:
                strong_connect(target)
                low_links[node_id] = min(low_links[node_id], low_links[target])
            elif target in on_stack:
                low_links[node_id] = min(low_links[node_id], indices[target])
        if low_links[node_id] == indices[node_id]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node_id:
                    break
            is_self_loop = node_id in dependencies[node_id]
            is_cycle = len(component) > 1 or is_self_loop
            # A cycle is only un-restorable when it has no mutable node where
            # it can be broken: mutable shells are built first and back-edges
            # are patched in afterwards. A cycle made entirely of frozen items
            # cannot be constructed at all.
            if is_cycle and all(member in frozen_nodes for member in component):
                raise GraphRestorationError(
                    "Cannot restore a reference cycle that includes a frozen item: "
                    + ", ".join(sorted(component))
                )

    for node_id in parsed:
        if node_id not in indices:
            strong_connect(node_id)

    # All checks passed: construction below never needs to bail out.
    built: dict[str, object] = {}
    pending_mutable: list[str] = []

    def build(node_id: str) -> object:
        if node_id in built:
            return built[node_id]
        type_name, fields = parsed[node_id]
        if type_name in ("dict", "list"):
            instance: object = [] if type_name == "list" else {}
            built[node_id] = instance
            pending_mutable.append(node_id)
            return instance

        item_class, is_frozen = registry[type_name]
        if is_frozen:
            # The frozen subgraph is a DAG (frozen cycles were rejected), so
            # every dependency can be fully built before this item's init.
            for target in dependencies[node_id]:
                build(target)
            resolved = {
                _decode_key(key): _resolve_ref(value, built)
                for key, value in fields.items()
            }
            instance = item_class(**resolved)
            built[node_id] = instance
            return instance

        instance = _create_shell(item_class)
        built[node_id] = instance
        pending_mutable.append(node_id)
        return instance

    # Mutable shells are created first so cycles can point at them; frozen
    # items are built on demand from already-resolved dependencies.
    for node_id in parsed:
        build(node_id)

    for node_id in pending_mutable:
        type_name, fields = parsed[node_id]
        target = built[node_id]
        if type_name == "list":
            resolved_list = _resolve_ref(fields, built)
            assert isinstance(resolved_list, list)
            cast_target = cast("list[object]", target)
            cast_target.extend(resolved_list)
        elif type_name == "dict":
            cast_target = cast("dict[str, object]", target)
            for key, value in fields.items():
                cast_target[_decode_key(key)] = _resolve_ref(value, built)
        else:
            adapter = ItemAdapter(target)
            for key, value in fields.items():
                adapter[_decode_key(key)] = _resolve_ref(value, built)

    return built[root_id]
