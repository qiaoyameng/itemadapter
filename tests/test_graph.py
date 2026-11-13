from __future__ import annotations

import json
from dataclasses import dataclass

import attrs
import pytest

from itemadapter import (
    GRAPH_FORMAT_VERSION,
    GraphFormatError,
    GraphRestorationError,
    ItemAdapter,
    build_item_registry,
    graph_from_item,
    item_from_graph,
)
from itemadapter._graph import _REF_KEY


@dataclass
class Child:
    name: str
    back: object = None


@dataclass
class Parent:
    child: Child
    other: Child | None = None


@dataclass(frozen=True)
class FrozenNode:
    name: str
    peer: object = None


@attrs.define
class AttrsNode:
    name: str
    peer: object = None


@attrs.define(frozen=True)
class FrozenAttrsNode:
    name: str
    peer: object = None


@pytest.fixture
def registry() -> dict:
    return build_item_registry([Parent, Child, FrozenNode, AttrsNode, FrozenAttrsNode])


class TestExport:
    def test_document_envelope(self):
        document = graph_from_item(Child(name="x"))
        assert document["version"] is GRAPH_FORMAT_VERSION
        assert isinstance(document["nodes"], list)
        assert document["root"] == {_REF_KEY: "0"}
        # The document is plain JSON data.
        json.dumps(document)

    def test_shared_child_keeps_identity(self, registry):
        shared = Child(name="shared")
        root = Parent(child=shared, other=shared)
        document = graph_from_item(root)

        child_nodes = [
            node for node in document["nodes"] if node["type"] == "Child"
        ]
        assert len(child_nodes) == 1
        parent_node = next(node for node in document["nodes"] if node["type"] == "Parent")
        assert parent_node["fields"]["child"] == parent_node["fields"]["other"]

        restored = item_from_graph(document, registry)
        assert restored.child is restored.other
        assert restored.child.name == "shared"

    def test_list_back_edge_cycle(self, registry):
        items: list = []
        child = Child(name="loopy", back=items)
        items.append(child)
        root = Child(name="root", back=items)

        document = graph_from_item(root)
        restored = item_from_graph(document, registry)

        assert isinstance(restored.back, list)
        assert restored.back[0].back is restored.back
        assert restored.back[0].name == "loopy"

    def test_item_self_cycle(self, registry):
        node = AttrsNode(name="self")
        node.peer = node
        restored = item_from_graph(graph_from_item(node), registry)
        assert restored.peer is restored

    def test_shared_list_and_dict_identity(self):
        shared_list = [1, 2]
        shared_dict = {"k": 3}
        root = {"a": shared_list, "b": shared_list, "c": shared_dict, "d": shared_dict}
        restored = item_from_graph(graph_from_item(root), {})
        assert restored["a"] is restored["b"]
        assert restored["c"] is restored["d"]

    def test_json_round_trip(self, registry):
        root = Parent(Child(name="a"), Child(name="b"))
        document = json.loads(json.dumps(graph_from_item(root)))
        restored = item_from_graph(document, registry)
        assert restored.child.name == "a"
        assert restored.other.name == "b"

    def test_dollar_prefixed_key_is_escaped(self, registry):
        root = Child(name="d", back={_REF_KEY: "literal", "plain": [1, {"k": 2}]})
        restored = item_from_graph(graph_from_item(root), registry)
        assert restored.back[_REF_KEY] == "literal"
        assert restored.back["plain"] == [1, {"k": 2}]

    def test_non_string_key_rejected(self):
        with pytest.raises(GraphFormatError):
            graph_from_item(Child(name="x", back={1: "no"}))

    def test_tuple_and_set_rejected(self):
        with pytest.raises(GraphFormatError):
            graph_from_item(Child(name="x", back=(1, 2)))
        with pytest.raises(GraphFormatError):
            graph_from_item(Child(name="x", back={1, 2}))

    def test_unsupported_scalar_rejected(self):
        with pytest.raises(GraphFormatError):
            graph_from_item(Child(name="x", back=object()))

    def test_non_item_root_rejected(self):
        with pytest.raises(GraphFormatError):
            graph_from_item([1, 2])

    def test_asdict_still_copies_shared_items(self):
        # Ordinary asdict behavior is intentionally untouched: shared items are
        # duplicated and cycles are not handled.
        shared = Child(name="shared")
        assert ItemAdapter(Parent(shared, shared)).asdict() == {
            "child": {"name": "shared", "back": None},
            "other": {"name": "shared", "back": None},
        }


class TestRegistry:
    def test_class_name_registry(self):
        registry = build_item_registry([Child, Parent])
        assert set(registry) == {"Child", "Parent"}
        assert registry["Child"] == (Child, False)

    def test_explicit_name_mapping(self):
        registry = build_item_registry({"my_child": Child})
        assert registry["my_child"] == (Child, False)

    def test_duplicate_name_rejected(self):
        with pytest.raises(GraphFormatError):
            build_item_registry([type("Same", (), {}), type("Same", (), {})])
        # Explicit names resolve the collision.
        assert build_item_registry({"first": Child, "second": Parent})

    def test_forced_frozen(self):
        registry = build_item_registry([Child], frozen=[Child])
        assert registry["Child"][1] is True

    def test_frozen_detection(self):
        assert build_item_registry([FrozenNode])["FrozenNode"][1] is True
        assert build_item_registry([Child])["Child"][1] is False
        assert build_item_registry([FrozenAttrsNode])["FrozenAttrsNode"][1] is True
        assert build_item_registry([AttrsNode])["AttrsNode"][1] is False

    def test_non_class_entry_rejected(self):
        with pytest.raises(GraphFormatError):
            build_item_registry([1])  # type: ignore[list-item]


class TestRestoreValidation:
    def _node(self, node_id: str, type_name: str, fields: object) -> dict:
        return {"id": node_id, "type": type_name, "fields": fields}

    def test_duplicate_id_rejected_before_construction(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node("0", "Child", {"name": "a", "back": None}),
                self._node("0", "Child", {"name": "b", "back": None}),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="Duplicate node id"):
            item_from_graph(document, registry)

    def test_dangling_reference_rejected_and_atomic(self):
        constructed = []

        @dataclass
        class Tracked:
            value: object

            def __post_init__(self) -> None:
                constructed.append(self)

        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node("0", "Tracked", {"value": {_REF_KEY: "missing"}}),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="Dangling reference"):
            item_from_graph(document, build_item_registry([Tracked]))
        assert constructed == []

    def test_nested_dangling_reference(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node("0", "Child", {"name": "x", "back": {"k": [{_REF_KEY: "99"}]}}),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="Dangling reference"):
            item_from_graph(document, registry)

    def test_unknown_type_rejected(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [self._node("0", "Ghost", {})],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="Unknown item type"):
            item_from_graph(document, registry)

    def test_node_budget_enforced(self, registry):
        document = graph_from_item(Parent(Child("a"), Child("b")))
        with pytest.raises(GraphRestorationError, match="max_nodes"):
            item_from_graph(document, registry, max_nodes=1)
        assert item_from_graph(document, registry, max_nodes=3) is not None

    def test_bad_version_rejected(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION + 1,
            "nodes": [],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="format version"):
            item_from_graph(document, registry)

    def test_root_must_reference_existing_node(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [],
            "root": {_REF_KEY: "42"},
        }
        with pytest.raises(GraphRestorationError):
            item_from_graph(document, registry)

    def test_payload_type_name_is_never_imported(self, registry):
        # A type name that is not in the registry must fail even though a
        # module-qualified name could otherwise be importable.
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node("0", "os.PathLike", {}),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="Unknown item type"):
            item_from_graph(document, registry)

    def test_malformed_documents(self, registry):
        with pytest.raises(GraphRestorationError):
            item_from_graph([], registry)
        with pytest.raises(GraphRestorationError):
            item_from_graph({"version": GRAPH_FORMAT_VERSION, "nodes": {}}, registry)

    def test_ref_marker_with_extra_keys_rejected(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node(
                    "0",
                    "Child",
                    {"name": "x", "back": {_REF_KEY: "0", "extra": 1}},
                ),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="marker"):
            item_from_graph(document, registry)

    def test_frozen_cycle_rejected_before_construction(self):
        constructed = []

        @dataclass(frozen=True)
        class TrackedFrozen:
            name: str
            peer: object = None

            def __post_init__(self) -> None:
                constructed.append(self)

        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node("0", "TrackedFrozen", {"name": "a", "peer": {_REF_KEY: "1"}}),
                self._node("1", "TrackedFrozen", {"name": "b", "peer": {_REF_KEY: "0"}}),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="frozen item"):
            item_from_graph(document, build_item_registry([TrackedFrozen]))
        assert constructed == []

    def test_frozen_self_loop_rejected(self, registry):
        document = {
            "version": GRAPH_FORMAT_VERSION,
            "nodes": [
                self._node("0", "FrozenNode", {"name": "a", "peer": {_REF_KEY: "0"}}),
            ],
            "root": {_REF_KEY: "0"},
        }
        with pytest.raises(GraphRestorationError, match="frozen item"):
            item_from_graph(document, registry)

    def test_frozen_dag_is_constructed(self, registry):
        root = FrozenNode(name="a", peer=FrozenNode(name="b"))
        restored = item_from_graph(graph_from_item(root), registry)
        assert isinstance(restored, FrozenNode)
        assert restored.peer.name == "b"

    def test_cycle_with_a_mutable_node_is_restored(self, registry):
        mutable = Child(name="m")
        frozen = FrozenNode(name="f", peer=mutable)
        mutable.back = frozen
        restored = item_from_graph(
            graph_from_item(frozen), build_item_registry([FrozenNode, Child])
        )
        assert isinstance(restored, FrozenNode)
        assert restored.peer.back is restored
