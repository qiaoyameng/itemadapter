from __future__ import annotations

import dataclasses

import pytest

from itemadapter import MISSING, ItemAdapter
from itemadapter._imports import attrs
from tests import requires_attrs


@dataclasses.dataclass(frozen=True)
class DataclassLeaf:
    number: int
    name: str = "leaf"


@dataclasses.dataclass(frozen=True)
class DataclassBranch:
    leaf: DataclassLeaf
    numbers: tuple
    words: list
    table: dict
    label: str = "branch"


@dataclasses.dataclass(frozen=True)
class DataclassTree:
    branch: DataclassBranch
    name: str = "tree"


@dataclasses.dataclass(frozen=True)
class ValidatedDataclass:
    number: int

    def __post_init__(self) -> None:
        if self.number < 0:
            raise ValueError("number must be non-negative")


@dataclasses.dataclass(frozen=True)
class DataclassWithProperty:
    name: str

    @property
    def shout(self) -> str:
        return self.name.upper()


@dataclasses.dataclass(frozen=True)
class DataclassWithInitFalse:
    number: int = 1
    excluded: int = dataclasses.field(init=False, default=100)


def make_dataclass_tree() -> tuple[DataclassTree, DataclassLeaf]:
    shared_leaf = DataclassLeaf(1)
    tree = DataclassTree(
        DataclassBranch(
            leaf=shared_leaf,
            numbers=(1, 2, 3),
            words=[shared_leaf, 9],
            table={"key": shared_leaf},
        )
    )
    return tree, shared_leaf


class TestDataclassCopyWith:
    def test_top_level_field_change(self):
        tree, _ = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(name="renamed")
        assert new_tree is not tree
        assert new_tree.name == "renamed"
        assert tree.name == "tree"
        # The untouched branch keeps its identity.
        assert new_tree.branch is tree.branch

    def test_deep_model_field_change(self):
        tree, shared_leaf = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            path_changes={("branch", "leaf", "number"): 42}
        )
        assert new_tree.branch.leaf.number == 42
        assert new_tree.branch.leaf.name == "leaf"
        # Only the affected branch is rebuilt.
        assert new_tree is not tree
        assert new_tree.branch is not tree.branch
        assert new_tree.branch.leaf is not shared_leaf
        assert new_tree.branch.numbers is tree.branch.numbers
        assert new_tree.branch.words is tree.branch.words
        assert new_tree.branch.table is tree.branch.table
        # The caller's object is untouched.
        assert tree.branch.leaf.number == 1
        assert tree.branch.leaf is shared_leaf

    def test_change_through_sequence_of_items(self):
        tree, shared_leaf = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            path_changes={("branch", "words", 0, "name"): "changed"}
        )
        assert new_tree.branch.words[0].name == "changed"
        assert new_tree.branch.words[0] is not shared_leaf
        # Other elements keep their identity, and the original is untouched.
        assert new_tree.branch.words[1] == 9
        assert tree.branch.words[0].name == "leaf"
        assert tree.branch.words[0] is shared_leaf

    def test_change_through_mapping_of_items(self):
        tree, shared_leaf = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            path_changes={("branch", "table", "key", "number"): 7}
        )
        assert new_tree.branch.table["key"].number == 7
        assert new_tree.branch.table["key"] is not shared_leaf
        assert tree.branch.table["key"] is shared_leaf

    def test_sequence_element_change_keeps_container_type(self):
        tree, _ = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            path_changes={("branch", "numbers", 1): 99}
        )
        assert isinstance(new_tree.branch.numbers, tuple)
        assert new_tree.branch.numbers == (1, 99, 3)

    def test_negative_index(self):
        tree, _ = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            path_changes={("branch", "numbers", -1): 0}
        )
        assert new_tree.branch.numbers == (1, 2, 0)

    def test_mapping_key_addition_and_deletion(self):
        tree, _ = make_dataclass_tree()
        added = ItemAdapter(tree).copy_with(
            path_changes={("branch", "table", "new_key"): 1}
        )
        assert added.branch.table["new_key"] == 1
        assert "new_key" not in tree.branch.table

        removed = ItemAdapter(tree).copy_with(
            path_changes={("branch", "table", "key"): MISSING}
        )
        assert "key" not in removed.branch.table
        assert "key" in tree.branch.table

    def test_sequence_element_deletion(self):
        tree, _ = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            path_changes={("branch", "words", 1): MISSING}
        )
        assert new_tree.branch.words == [tree.branch.words[0]]
        assert tree.branch.words == [tree.branch.words[0], 9]

    def test_multiple_changes_in_one_call(self):
        tree, _ = make_dataclass_tree()
        new_tree = ItemAdapter(tree).copy_with(
            name="root",
            path_changes={
                ("branch", "label"): "new-branch",
                ("branch", "leaf", "number"): 3,
            },
        )
        assert new_tree.name == "root"
        assert new_tree.branch.label == "new-branch"
        assert new_tree.branch.leaf.number == 3

    def test_shared_subobject_relations(self):
        shared_leaf = DataclassLeaf(1)

        @dataclasses.dataclass(frozen=True)
        class Node:
            left: DataclassLeaf
            right: DataclassLeaf
            refs: list

        node = Node(left=shared_leaf, right=shared_leaf, refs=[shared_leaf, shared_leaf])
        new_node = ItemAdapter(node).copy_with(
            path_changes={("left", "number"): 9}
        )
        assert new_node.left.number == 9
        # The other references to the shared object stay untouched.
        assert new_node.right is shared_leaf
        assert new_node.refs is node.refs
        assert all(ref is shared_leaf for ref in new_node.refs)
        assert node.left is shared_leaf

    def test_factory_is_not_called_again(self):
        created = []

        def factory() -> list:
            value = []
            created.append(value)
            return value

        @dataclasses.dataclass(frozen=True)
        class WithFactory:
            number: int = 1
            produced: list = dataclasses.field(default_factory=factory)

        item = WithFactory()
        assert len(created) == 1
        new_item = ItemAdapter(item).copy_with(number=2)
        assert len(created) == 1
        assert new_item.produced is item.produced

    def test_constructor_runs_on_each_rebuilt_level(self):
        calls: list[str] = []

        @dataclasses.dataclass(frozen=True)
        class CountedLeaf:
            number: int

            def __init__(self, number: int) -> None:
                calls.append("leaf")
                object.__setattr__(self, "number", number)

        @dataclasses.dataclass(frozen=True)
        class CountedBranch:
            leaf: CountedLeaf
            label: str = "branch"

            def __init__(self, leaf: CountedLeaf, label: str = "branch") -> None:
                calls.append("branch")
                object.__setattr__(self, "leaf", leaf)
                object.__setattr__(self, "label", label)

        branch = CountedBranch(CountedLeaf(1))
        calls.clear()
        ItemAdapter(branch).copy_with(path_changes={("leaf", "number"): 42})
        assert calls == ["leaf", "branch"]

    def test_constructor_validation_failure_keeps_original(self):
        item = ValidatedDataclass(1)
        with pytest.raises(ValueError, match="non-negative"):
            ItemAdapter(item).copy_with(number=-1)
        assert item.number == 1

    def test_nested_constructor_validation_failure_keeps_original(self):
        tree, shared_leaf = make_dataclass_tree()

        @dataclasses.dataclass(frozen=True)
        class ValidatedLeaf:
            number: int

            def __post_init__(self) -> None:
                if self.number < 0:
                    raise ValueError("negative leaf")

        @dataclasses.dataclass(frozen=True)
        class Holder:
            leaf: ValidatedLeaf
            sibling: str = "keep"

        holder = Holder(ValidatedLeaf(5), "keep")
        with pytest.raises(ValueError, match="negative leaf"):
            ItemAdapter(holder).copy_with(path_changes={("leaf", "number"): -1})
        assert holder.leaf.number == 5
        assert holder.sibling == "keep"

    def test_unknown_field_raises_keyerror(self):
        tree, _ = make_dataclass_tree()
        with pytest.raises(KeyError, match="DataclassTree does not support field: nope"):
            ItemAdapter(tree).copy_with(nope=1)
        with pytest.raises(KeyError, match="DataclassLeaf does not support field: nope"):
            ItemAdapter(tree).copy_with(path_changes={("branch", "leaf", "nope"): 1})

    def test_read_only_property_raises_keyerror(self):
        item = DataclassWithProperty("name")
        with pytest.raises(KeyError, match="read-only attribute"):
            ItemAdapter(item).copy_with(shout="X")
        with pytest.raises(KeyError, match="read-only attribute"):
            ItemAdapter(item).copy_with(path_changes={("shout",): "X"})

    def test_field_deletion_is_illegal(self):
        tree, _ = make_dataclass_tree()
        with pytest.raises(KeyError, match="cannot be deleted"):
            ItemAdapter(tree).copy_with(name=MISSING)
        with pytest.raises(KeyError, match="cannot be deleted"):
            ItemAdapter(tree).copy_with(path_changes={("branch", "leaf"): MISSING})

    def test_init_false_field_cannot_be_rebuilt(self):
        item = DataclassWithInitFalse()
        with pytest.raises(TypeError, match="init=False"):
            ItemAdapter(item).copy_with(excluded=2)

    def test_no_changes_returns_same_object(self):
        tree, _ = make_dataclass_tree()
        assert ItemAdapter(tree).copy_with() is tree


@requires_attrs
def _build_attrs_classes():
    @attrs.frozen
    class AttrsLeaf:
        number: int
        name: str = "leaf"

    @attrs.frozen
    class AttrsBranch:
        leaf: AttrsLeaf
        numbers: tuple = ()
        words: list = attrs.field(factory=list)
        table: dict = attrs.field(factory=dict)
        label: str = "branch"

    @attrs.frozen
    class AttrsTree:
        branch: AttrsBranch
        name: str = "tree"

    return AttrsLeaf, AttrsBranch, AttrsTree


@requires_attrs
class TestAttrsCopyWith:
    def test_top_level_and_deep_changes(self):
        AttrsLeaf, AttrsBranch, AttrsTree = _build_attrs_classes()
        shared_leaf = AttrsLeaf(1)
        tree = AttrsTree(
            AttrsBranch(shared_leaf, (1, 2, 3), [shared_leaf, 9], {"key": shared_leaf})
        )
        new_tree = ItemAdapter(tree).copy_with(
            name="renamed",
            path_changes={("branch", "leaf", "number"): 42},
        )
        assert new_tree.name == "renamed"
        assert new_tree.branch.leaf.number == 42
        assert new_tree.branch is not tree.branch
        assert new_tree.branch.numbers is tree.branch.numbers
        assert new_tree.branch.words is tree.branch.words
        assert new_tree.branch.table is tree.branch.table
        assert tree.branch.leaf.number == 1

    def test_validator_runs_and_keeps_original(self):
        @attrs.frozen
        class Validated:
            number: int = attrs.field(validator=attrs.validators.instance_of(int))

        item = Validated(1)
        with pytest.raises(TypeError, match="'number' must be"):
            ItemAdapter(item).copy_with(number="bad")
        assert item.number == 1

    def test_factory_is_not_called_again(self):
        created = []

        def factory() -> list:
            value = []
            created.append(value)
            return value

        @attrs.frozen
        class WithFactory:
            number: int = 1
            produced: list = attrs.field(factory=factory)

        item = WithFactory()
        assert len(created) == 1
        new_item = ItemAdapter(item).copy_with(number=2)
        assert len(created) == 1
        assert new_item.produced is item.produced

    def test_private_attribute_uses_init_alias(self):
        @attrs.frozen
        class WithPrivate:
            _number: int = attrs.field(alias="number")
            label: str = "x"

        item = WithPrivate(number=1)
        new_item = ItemAdapter(item).copy_with(_number=5)
        assert new_item._number == 5
        assert item._number == 1

    def test_read_only_property_raises_keyerror(self):
        @attrs.frozen
        class WithProperty:
            name: str

            @property
            def shout(self) -> str:
                return self.name.upper()

        item = WithProperty("name")
        with pytest.raises(KeyError, match="read-only attribute"):
            ItemAdapter(item).copy_with(shout="X")

    def test_shared_subobject_relations(self):
        AttrsLeaf, _, _ = _build_attrs_classes()
        shared = AttrsLeaf(1)

        @attrs.frozen
        class Node:
            left: AttrsLeaf
            right: AttrsLeaf
            refs: list

        node = Node(left=shared, right=shared, refs=[shared])
        new_node = ItemAdapter(node).copy_with(path_changes={("left", "number"): 9})
        assert new_node.left.number == 9
        assert new_node.right is shared
        assert new_node.refs is node.refs
        assert node.left is shared

    def test_classic_attrs_api(self):
        import attr

        @attr.s(frozen=True)
        class Classic:
            x = attr.ib(type=int, default=1)
            y = attr.ib(type=int, default=2)

        item = Classic()
        new_item = ItemAdapter(item).copy_with(x=9)
        assert new_item is not item
        assert new_item.x == 9
        assert new_item.y == 2
        assert item.x == 1


class TestDictCopyWith:
    def test_nested_dict_and_sequence(self):
        item = {"a": {"b": [1, 2, 3]}, "kept": {"c": 1}}
        new_item = ItemAdapter(item).copy_with(path_changes={("a", "b", 1): 99})
        assert new_item == {"a": {"b": [1, 99, 3]}, "kept": {"c": 1}}
        assert item == {"a": {"b": [1, 2, 3]}, "kept": {"c": 1}}
        assert new_item["kept"] is item["kept"]
        assert new_item["a"] is not item["a"]

    def test_add_and_delete_keys(self):
        item = {"a": 1}
        added = ItemAdapter(item).copy_with(path_changes={("b",): 2})
        assert added == {"a": 1, "b": 2}
        removed = ItemAdapter(item).copy_with(path_changes={("a",): MISSING})
        assert removed == {}
        assert item == {"a": 1}

    def test_dict_subclass_keeps_type(self):
        class MyDict(dict):
            pass

        item = MyDict(a={"b": 1})
        new_item = ItemAdapter(item).copy_with(path_changes={("a", "b"): 2})
        assert isinstance(new_item, MyDict)
        assert isinstance(new_item["a"], dict)


class TestPathValidation:
    def test_set_and_frozenset_are_not_subscriptable(self):
        @dataclasses.dataclass(frozen=True)
        class Containers:
            bag: set
            frozen_bag: frozenset

        item = Containers({1}, frozenset({1}))
        with pytest.raises(TypeError, match="not subscriptable"):
            ItemAdapter(item).copy_with(path_changes={("bag", 0): 1})
        with pytest.raises(TypeError, match="not subscriptable"):
            ItemAdapter(item).copy_with(path_changes={("frozen_bag", 0): 1})

    def test_index_out_of_range(self):
        item = {"values": [1, 2]}
        with pytest.raises(IndexError, match="out of range"):
            ItemAdapter(item).copy_with(path_changes={("values", 5): 1})
        with pytest.raises(IndexError, match="out of range"):
            ItemAdapter(item).copy_with(path_changes={("values", -3): 1})

    def test_string_values_cannot_be_traversed(self):
        @dataclasses.dataclass(frozen=True)
        class WithString:
            name: str

        item = WithString("abc")
        with pytest.raises(TypeError, match="cannot be traversed"):
            ItemAdapter(item).copy_with(path_changes={("name", "x"): 1})

    def test_missing_mapping_key_on_intermediate_node(self):
        item = {"a": 1}
        with pytest.raises(KeyError, match="nope"):
            ItemAdapter(item).copy_with(path_changes={("nope", "deeper"): 1})
        with pytest.raises(KeyError, match="nope"):
            ItemAdapter(item).copy_with(path_changes={("nope",): MISSING})

    def test_invalid_path_shapes(self):
        item = {"a": 1}
        for bad_path in ["a", 42, (), (1.5,), (None,), (True,)]:
            with pytest.raises((TypeError, ValueError)):
                ItemAdapter(item).copy_with(path_changes={bad_path: 1})

    def test_integer_index_on_item_rejected(self):
        @dataclasses.dataclass(frozen=True)
        class Model:
            number: int

        item = Model(1)
        with pytest.raises(TypeError, match="do not support integer indexing"):
            ItemAdapter(item).copy_with(path_changes={(0,): 1})

    def test_duplicate_path_rejected(self):
        item = {"a": 1}
        with pytest.raises(ValueError, match="duplicate change"):
            ItemAdapter(item).copy_with(a=2, path_changes={("a",): 3})

    def test_validation_runs_before_any_rebuilding(self):
        @dataclasses.dataclass(frozen=True)
        class Leaf:
            number: int

        @dataclasses.dataclass(frozen=True)
        class Root:
            leaf: Leaf
            values: list

        values = [1, 2]
        root = Root(Leaf(1), values)
        with pytest.raises(IndexError):
            ItemAdapter(root).copy_with(
                path_changes={
                    ("values", 9): 1,
                    ("leaf", "number"): 2,
                }
            )
        assert root.leaf.number == 1
        assert root.values is values


class TestMixedNesting:
    def test_item_dict_list_list_item_path(self):
        @dataclasses.dataclass(frozen=True)
        class Point:
            x: int
            y: int

        @dataclasses.dataclass(frozen=True)
        class Scene:
            layers: dict

        scene = Scene(layers={"points": [[Point(1, 2)], [Point(3, 4)]]})
        new_scene = ItemAdapter(scene).copy_with(
            path_changes={("layers", "points", 1, 0, "y"): 99}
        )
        assert new_scene.layers["points"][1][0].y == 99
        assert scene.layers["points"][1][0].y == 4
        # Untouched branches keep their identity.
        assert new_scene.layers["points"][0] is scene.layers["points"][0]
        assert new_scene.layers["points"][0][0] is scene.layers["points"][0][0]

    def test_changes_in_separate_branches(self):
        @dataclasses.dataclass(frozen=True)
        class Point:
            x: int
            y: int

        @dataclasses.dataclass(frozen=True)
        class Triple:
            a: Point
            b: Point
            c: Point

        p1, p2, p3 = Point(1, 1), Point(2, 2), Point(3, 3)
        triple = Triple(p1, p2, p3)
        new_triple = ItemAdapter(triple).copy_with(
            path_changes={("a", "x"): 10, ("c", "y"): 30}
        )
        assert new_triple.a is not p1
        assert new_triple.c is not p3
        assert new_triple.b is p2
        assert (new_triple.a.x, new_triple.c.y) == (10, 30)


class TestNonFrozenItems:
    def test_non_frozen_dataclass_is_not_mutated(self):
        @dataclasses.dataclass
        class Mutable:
            number: int
            values: list

        item = Mutable(1, [10, 20])
        new_item = ItemAdapter(item).copy_with(path_changes={("values", 0): 99})
        assert item.values == [10, 20]
        assert new_item.values == [99, 20]
