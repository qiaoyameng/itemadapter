from __future__ import annotations

from dataclasses import dataclass, field

import attrs
import pytest

from itemadapter import ItemAdapter


@dataclass(frozen=True)
class FrozenInner:
    x: int
    tags: list = field(default_factory=list)


@dataclass(frozen=True)
class FrozenOuter:
    inner: FrozenInner
    other: FrozenInner
    name: str = "outer"


@attrs.frozen
class AttrsInner:
    x: int = attrs.field(validator=attrs.validators.gt(0))


@attrs.frozen
class AttrsOuter:
    inner: AttrsInner = attrs.field()
    other: AttrsInner = attrs.field()


class TestFrozenDataclass:
    def test_deep_set_preserves_shared_branches(self):
        shared = FrozenInner(1, ["a"])
        outer = FrozenOuter(shared, shared)
        new = ItemAdapter(outer).copy_with(("inner", "x"), 2)
        assert new is not outer
        assert new.inner is not shared
        assert new.inner.x == 2
        assert new.inner.tags is shared.tags
        assert new.other is shared
        assert new.name == "outer"
        assert outer.inner.x == 1
        assert outer.other is shared

    def test_top_level_set(self):
        outer = FrozenOuter(FrozenInner(1), FrozenInner(2))
        new = ItemAdapter(outer).copy_with(("name",), "new")
        assert new.name == "new"
        assert new.inner is outer.inner
        assert outer.name == "outer"

    def test_original_not_mutated_on_mutable_item(self):
        @dataclass
        class Mutable:
            x: int

        obj = Mutable(1)
        new = ItemAdapter(obj).copy_with(("x",), 2)
        assert new.x == 2
        assert obj.x == 1


class TestFrozenAttrs:
    def test_deep_set_preserves_shared_branches(self):
        shared = AttrsInner(1)
        outer = AttrsOuter(shared, shared)
        new = ItemAdapter(outer).copy_with(("inner", "x"), 2)
        assert new is not outer
        assert new.inner is not shared
        assert new.inner.x == 2
        assert new.other is shared
        assert outer.inner.x == 1

    def test_validator_runs_and_failure_is_atomic(self):
        outer = AttrsOuter(AttrsInner(1), AttrsInner(2))
        with pytest.raises(ValueError, match="must be > 0"):
            ItemAdapter(outer).copy_with(("inner", "x"), -1)
        assert outer.inner.x == 1
        assert ItemAdapter(outer)["inner"].x == 1


class TestCountingFactory:
    def test_dataclass_default_factory_not_re_run(self):
        calls = []

        def factory():
            calls.append(1)
            return []

        @dataclass(frozen=True)
        class WithFactory:
            inner: FrozenInner
            items: list = field(default_factory=factory)

        obj = WithFactory(FrozenInner(1))
        assert len(calls) == 1
        new = ItemAdapter(obj).copy_with(("inner", "x"), 2)
        assert len(calls) == 1
        assert new.items is obj.items
        assert new.inner.x == 2

    def test_attrs_factory_not_re_run(self):
        calls = []

        def factory():
            calls.append(1)
            return []

        @attrs.frozen
        class AttrsWithFactory:
            inner: AttrsInner = attrs.field()
            items: list = attrs.field(factory=factory)

        obj = AttrsWithFactory(AttrsInner(1))
        assert len(calls) == 1
        new = ItemAdapter(obj).copy_with(("inner", "x"), 2)
        assert len(calls) == 1
        assert new.items is obj.items


class TestNestedPaths:
    def test_through_dict_and_sequence(self):
        @dataclass(frozen=True)
        class Doc:
            meta: dict
            rows: tuple

        doc = Doc(meta={"a": {"b": 1}}, rows=([1, 2], [3]))
        new = ItemAdapter(doc).copy_with(("meta", "a", "b"), 42)
        assert new.meta["a"]["b"] == 42
        assert doc.meta["a"]["b"] == 1
        assert new.meta is not doc.meta

        new2 = ItemAdapter(doc).copy_with(("rows", 0, 1), 99)
        assert new2.rows[0] == [1, 99]
        assert new2.rows[1] is doc.rows[1]
        assert doc.rows[0] == [1, 2]

    def test_negative_index(self):
        adapter = ItemAdapter({"items": [1, 2, 3]})
        new = adapter.copy_with(("items", -1), 30)
        assert new == {"items": [1, 2, 30]}
        assert adapter.item == {"items": [1, 2, 3]}

    def test_new_dict_key(self):
        adapter = ItemAdapter({"a": 1})
        new = adapter.copy_with(("b",), 2)
        assert new == {"a": 1, "b": 2}
        assert adapter.item == {"a": 1}

    def test_top_level_dict(self):
        adapter = ItemAdapter({"nested": {"x": 1}})
        new = adapter.copy_with(("nested", "x"), 2)
        assert new == {"nested": {"x": 2}}
        assert adapter.item == {"nested": {"x": 1}}


class TestDeletion:
    def test_dict_key(self):
        adapter = ItemAdapter({"a": {"b": 1, "c": 2}})
        new = adapter.copy_with(("a", "b"))
        assert new == {"a": {"c": 2}}
        assert adapter.item == {"a": {"b": 1, "c": 2}}

    def test_missing_dict_key(self):
        adapter = ItemAdapter({"a": {}})
        with pytest.raises(KeyError):
            adapter.copy_with(("a", "b"))

    def test_dataclass_field_is_illegal(self):
        outer = FrozenOuter(FrozenInner(1), FrozenInner(2))
        with pytest.raises(TypeError, match="cannot delete field"):
            ItemAdapter(outer).copy_with(("name",))

    def test_attrs_field_is_illegal(self):
        outer = AttrsOuter(AttrsInner(1), AttrsInner(2))
        with pytest.raises(TypeError, match="cannot delete field"):
            ItemAdapter(outer).copy_with(("inner",))

    def test_sequence_element_is_illegal(self):
        adapter = ItemAdapter({"items": [1, 2]})
        with pytest.raises(TypeError, match="cannot delete elements"):
            adapter.copy_with(("items", 0))


class TestInvalidPaths:
    def test_unknown_field(self):
        outer = FrozenOuter(FrozenInner(1), FrozenInner(2))
        with pytest.raises(KeyError, match="FrozenOuter does not support field: unknown"):
            ItemAdapter(outer).copy_with(("unknown",), 1)

    def test_unknown_nested_field(self):
        outer = FrozenOuter(FrozenInner(1), FrozenInner(2))
        with pytest.raises(KeyError, match="FrozenInner does not support field: unknown"):
            ItemAdapter(outer).copy_with(("inner", "unknown"), 1)

    def test_read_only_property(self):
        @dataclass(frozen=True)
        class WithProperty:
            x: int = 1

            @property
            def double(self):
                return self.x * 2

        with pytest.raises(TypeError, match="read-only"):
            ItemAdapter(WithProperty()).copy_with(("double",), 4)

    def test_string_path(self):
        adapter = ItemAdapter({"a": 1})
        with pytest.raises(TypeError, match="iterable of strings and integers"):
            adapter.copy_with("a", 2)

    def test_empty_path(self):
        adapter = ItemAdapter({"a": 1})
        with pytest.raises(ValueError, match="must not be empty"):
            adapter.copy_with((), 2)

    def test_invalid_step_type(self):
        adapter = ItemAdapter({"a": 1})
        with pytest.raises(TypeError, match="strings or integers"):
            adapter.copy_with((1.5,), 2)

    def test_bool_index(self):
        adapter = ItemAdapter({"items": [1]})
        with pytest.raises(TypeError, match="strings or integers"):
            adapter.copy_with(("items", True), 2)

    def test_index_out_of_range(self):
        adapter = ItemAdapter({"items": [1]})
        with pytest.raises(IndexError, match="out of range"):
            adapter.copy_with(("items", 5), 2)

    def test_string_into_sequence(self):
        adapter = ItemAdapter({"items": [1]})
        with pytest.raises(TypeError, match="as an index"):
            adapter.copy_with(("items", "x"), 2)

    def test_int_into_item(self):
        outer = FrozenOuter(FrozenInner(1), FrozenInner(2))
        with pytest.raises(TypeError, match="as a field name"):
            ItemAdapter(outer).copy_with((0,), 1)

    def test_traverse_into_scalar(self):
        adapter = ItemAdapter({"a": 1})
        with pytest.raises(TypeError, match="cannot traverse"):
            adapter.copy_with(("a", "b"), 2)


class TestConstructorValidation:
    def test_dataclass_post_init_failure_is_atomic(self):
        @dataclass(frozen=True)
        class Validated:
            x: int

            def __post_init__(self):
                if self.x < 0:
                    raise ValueError("x must be >= 0")

        obj = Validated(1)
        with pytest.raises(ValueError, match="x must be >= 0"):
            ItemAdapter(obj).copy_with(("x",), -5)
        assert obj.x == 1

    def test_nested_validation_failure_keeps_original(self):
        outer = AttrsOuter(AttrsInner(1), AttrsInner(2))
        with pytest.raises(ValueError):
            ItemAdapter(outer).copy_with(("other", "x"), 0)
        assert outer.other.x == 2
