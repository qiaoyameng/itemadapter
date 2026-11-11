from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, TypedDict

import pytest

from itemadapter._json_schema import apply_aliases_to_json_schema
from itemadapter.adapter import ItemAdapter
from tests import requires_attrs
from tests.test_json_schema import check_schemas

try:
    import attrs
except ImportError:
    attrs = None


@dataclass
class DataclassProduct:
    product_id: str = field(metadata={"alias": "sku"})
    name: str = "default-name"
    price: float = 0.0


@dataclass
class DataclassTwoFields:
    product_id: str = ""
    sku: str = ""


@dataclass
class DataclassNested:
    outer_id: str = field(metadata={"alias": "outer_sku"})
    inner: DataclassProduct | None = None


if attrs is not None:

    @attrs.define
    class AttrsProduct:
        product_id: str = attrs.field(metadata={"alias": "sku"})
        name: str = "default-name"
        price: float = 0.0

    @attrs.define
    class AttrsNested:
        outer_id: str = attrs.field(metadata={"alias": "outer_sku"})
        inner: AttrsProduct | None = None


class TypedDictProduct(TypedDict):
    product_id: Annotated[str, {"alias": "sku"}]
    name: str


class TestDataclassAlias:
    item_class = DataclassProduct

    def test_metadata_alias_get(self):
        adapter = ItemAdapter(self.item_class(product_id="A1"))
        assert adapter["sku"] == "A1"
        with pytest.raises(KeyError) as exc_info:
            adapter["product_id"]
        assert exc_info.value.args == ("product_id",)

    def test_metadata_alias_set_does_not_rename_underlying_field(self):
        item = self.item_class(product_id="A1")
        adapter = ItemAdapter(item)
        adapter["sku"] = "B2"
        assert item.product_id == "B2"
        assert not hasattr(item, "sku")

    def test_metadata_alias_delete(self):
        item = self.item_class(product_id="A1")
        adapter = ItemAdapter(item)
        del adapter["sku"]
        assert not hasattr(item, "product_id")

    def test_field_names_use_alias_namespace(self):
        adapter = ItemAdapter(self.item_class(product_id="A1"))
        assert sorted(adapter.field_names()) == ["name", "price", "sku"]
        assert sorted(iter(adapter)) == ["name", "price", "sku"]
        assert "sku" in adapter
        assert "product_id" not in adapter

    def test_asdict_uses_alias_namespace(self):
        adapter = ItemAdapter(self.item_class(product_id="A1", name="n", price=9.5))
        assert adapter.asdict() == {"sku": "A1", "name": "n", "price": 9.5}
        assert dict(adapter) == {"sku": "A1", "name": "n", "price": 9.5}

    def test_get_field_meta_through_alias(self):
        adapter = ItemAdapter(self.item_class(product_id="A1"))
        assert adapter.get_field_meta("sku") == MappingProxyType({"alias": "sku"})

    def test_caller_mapping_overrides_metadata(self):
        item = self.item_class(product_id="A1")
        adapter = ItemAdapter(item, field_aliases={"product_id": "external_sku"})
        assert adapter["external_sku"] == "A1"
        with pytest.raises(KeyError):
            adapter["sku"]
        assert sorted(adapter.field_names()) == ["external_sku", "name", "price"]

    def test_caller_mapping_without_metadata(self):
        @dataclass
        class Plain:
            product_id: str = ""
            name: str = ""

        adapter = ItemAdapter(Plain(product_id="A1"), field_aliases={"product_id": "sku"})
        assert adapter["sku"] == "A1"
        assert adapter.asdict() == {"sku": "A1", "name": ""}

    def test_default_adapter_keeps_original_names(self):
        @dataclass
        class Plain:
            product_id: str = ""
            name: str = ""

        adapter = ItemAdapter(Plain(product_id="A1"))
        assert adapter["product_id"] == "A1"
        assert sorted(adapter.field_names()) == ["name", "product_id"]
        assert adapter.asdict() == {"product_id": "A1", "name": ""}

    def test_unknown_field_rejected_before_access(self):
        item = self.item_class(product_id="A1")
        with pytest.raises(ValueError, match="unknown field 'nope'"):
            ItemAdapter(item, field_aliases={"nope": "x"})

    def test_duplicate_alias_rejected(self):
        @dataclass
        class TwoFields:
            a: str = ""
            b: str = ""

        with pytest.raises(ValueError, match="Duplicate alias 'x'"):
            ItemAdapter(TwoFields(), field_aliases={"a": "x", "b": "x"})

    def test_duplicate_metadata_alias_rejected(self):
        @dataclass
        class TwoMetadataFields:
            a: str = field(default="", metadata={"alias": "x"})
            b: str = field(default="", metadata={"alias": "x"})

        with pytest.raises(ValueError, match="Duplicate alias 'x'"):
            ItemAdapter(TwoMetadataFields())

    def test_alias_collision_with_unrenamed_field_rejected(self):
        with pytest.raises(ValueError, match="collides with the name of an unrenamed field"):
            ItemAdapter(
                DataclassTwoFields(),
                field_aliases={"product_id": "sku"},
            )

    def test_swap_aliases_allowed(self):
        item = DataclassTwoFields(product_id="p", sku="s")
        adapter = ItemAdapter(
            item,
            field_aliases={"product_id": "sku", "sku": "product_id"},
        )
        assert adapter["sku"] == "p"
        assert adapter["product_id"] == "s"
        assert adapter.asdict() == {"sku": "p", "product_id": "s"}
        assert sorted(adapter.field_names()) == ["product_id", "sku"]

    def test_swap_aliases_schema(self):
        schema = ItemAdapter.get_json_schema(
            DataclassTwoFields,
        )
        aliases = {"product_id": "sku", "sku": "product_id"}
        renamed = apply_aliases_to_json_schema(
            schema,
            ["product_id", "sku"],
            aliases,
            item_class_name=DataclassTwoFields.__name__,
        )
        check_schemas(
            renamed,
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sku": {"type": "string", "default": ""},
                    "product_id": {"type": "string", "default": ""},
                },
            },
        )

    def test_json_schema_uses_alias_namespace(self):
        schema = ItemAdapter.get_json_schema(self.item_class)
        check_schemas(
            schema,
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sku": {"type": "string"},
                    "name": {"type": "string", "default": "default-name"},
                    "price": {"type": "number", "default": 0.0},
                },
                "required": ["sku"],
            },
        )

    def test_nested_asdict_applies_each_mapping(self):
        item = DataclassNested(
            outer_id="O1",
            inner=DataclassProduct(product_id="I1", name="n"),
        )
        adapter = ItemAdapter(item)
        assert adapter.asdict() == {
            "outer_sku": "O1",
            "inner": {"sku": "I1", "name": "n", "price": 0.0},
        }

    def test_nested_json_schema_applies_each_mapping(self):
        schema = ItemAdapter.get_json_schema(DataclassNested)
        check_schemas(
            schema,
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outer_sku": {"type": "string"},
                    "inner": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "sku": {"type": "string"},
                                    "name": {
                                        "type": "string",
                                        "default": "default-name",
                                    },
                                    "price": {"type": "number", "default": 0.0},
                                },
                                "required": ["sku"],
                            },
                        ],
                        "default": None,
                    },
                },
                "required": ["outer_sku"],
            },
        )

    def test_failure_does_not_read_or_modify_item(self):
        @dataclass
        class Guarded:
            a: str = field(metadata={"alias": "x"})
            b: str = ""

        item = Guarded(a="1", b="2")
        original_repr = repr(item)
        with pytest.raises(ValueError):
            ItemAdapter(item, field_aliases={"a": "z", "b": "z"})
        with pytest.raises(ValueError):
            ItemAdapter(item, field_aliases={"unknown": "z"})
        assert repr(item) == original_repr

    def test_failed_operations_do_not_modify_item(self):
        item = DataclassTwoFields(product_id="p", sku="s")
        adapter = ItemAdapter(
            item,
            field_aliases={"product_id": "sku", "sku": "product_id"},
        )
        with pytest.raises(KeyError):
            adapter["nope"]
        with pytest.raises(KeyError):
            adapter["nope"] = "v"
        with pytest.raises(KeyError):
            del adapter["nope"]
        assert item.product_id == "p"
        assert item.sku == "s"


@requires_attrs
class TestAttrsAlias(TestDataclassAlias):
    item_class = AttrsProduct

    def test_nested_asdict_applies_each_mapping(self):
        item = AttrsNested(
            outer_id="O1",
            inner=AttrsProduct(product_id="I1", name="n"),
        )
        adapter = ItemAdapter(item)
        assert adapter.asdict() == {
            "outer_sku": "O1",
            "inner": {"sku": "I1", "name": "n", "price": 0.0},
        }

    def test_nested_json_schema_applies_each_mapping(self):
        schema = ItemAdapter.get_json_schema(AttrsNested)
        check_schemas(
            schema,
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outer_sku": {"type": "string"},
                    "inner": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "sku": {"type": "string"},
                                    "name": {
                                        "type": "string",
                                        "default": "default-name",
                                    },
                                    "price": {"type": "number", "default": 0.0},
                                },
                                "required": ["sku"],
                            },
                        ],
                        "default": None,
                    },
                },
                "required": ["outer_sku"],
            },
        )


class TestDictAlias:
    def test_caller_mapping_rejected_for_plain_dict(self):
        with pytest.raises(ValueError, match="does not define fields"):
            ItemAdapter({"product_id": "A1"}, field_aliases={"product_id": "sku"})

    def test_plain_dict_keeps_original_names(self):
        adapter = ItemAdapter({"product_id": "A1", "name": "n"})
        assert adapter["product_id"] == "A1"
        assert adapter.asdict() == {"product_id": "A1", "name": "n"}

    def test_typeddict_schema_uses_metadata_alias(self):
        schema = ItemAdapter.get_json_schema(TypedDictProduct)
        check_schemas(
            schema,
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sku": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["sku", "name"],
            },
        )

    def test_typeddict_instance_remains_plain_dict(self):
        item: TypedDictProduct = {"product_id": "A1", "name": "n"}
        adapter = ItemAdapter(item)
        assert adapter["product_id"] == "A1"
        with pytest.raises(ValueError, match="does not define fields"):
            ItemAdapter(item, field_aliases={"product_id": "sku"})


class TestAliasConsistencyAcrossAdapters:
    def test_original_name_never_accessible_when_aliased(self):
        @dataclass
        class DataClassItem:
            product_id: str = field(metadata={"alias": "sku"})
            name: str = ""

        @attrs.define
        class AttrsItemLocal:
            product_id: str = attrs.field(metadata={"alias": "sku"})
            name: str = ""

        for item in (DataClassItem(product_id="A1"), AttrsItemLocal(product_id="A1")):
            adapter = ItemAdapter(item)
            assert "sku" in adapter
            assert "product_id" not in adapter
            with pytest.raises(KeyError):
                adapter["product_id"]
            assert adapter.get("product_id") is None
            assert sorted(adapter.field_names()) == ["name", "sku"]


@requires_attrs
class TestFailureSafety:
    def test_invalid_aliases_never_invoke_field_getters(self):
        @attrs.define
        class Guarded:
            a: str
            b: str

            def __getattribute__(self, name):
                if name in ("a", "b"):
                    raise AssertionError(f"getter invoked for {name}")
                return object.__getattribute__(self, name)

        item = Guarded("1", "2")
        with pytest.raises(ValueError):
            ItemAdapter(item, field_aliases={"a": "x", "b": "x"})
        with pytest.raises(ValueError):
            ItemAdapter(item, field_aliases={"unknown": "x"})
        with pytest.raises(ValueError):
            ItemAdapter(item, field_aliases={"a": "b"})

    def test_original_name_access_never_involves_getter(self):
        @attrs.define
        class Guarded:
            a: str
            b: str

            def __getattribute__(self, name):
                if name in ("a", "b"):
                    raise AssertionError(f"getter invoked for {name}")
                return object.__getattribute__(self, name)

        adapter = ItemAdapter(Guarded("1", "2"), field_aliases={"a": "alpha"})
        with pytest.raises(KeyError):
            adapter["a"]
        with pytest.raises(KeyError):
            adapter["missing"]

    def test_failed_write_and_delete_leave_item_untouched(self):
        item = DataclassTwoFields(product_id="p", sku="s")
        adapter = ItemAdapter(
            item,
            field_aliases={"product_id": "sku", "sku": "product_id"},
        )
        with pytest.raises(KeyError):
            adapter["unknown"] = "v"
        with pytest.raises(KeyError):
            del adapter["unknown"]
        assert item.product_id == "p"
        assert item.sku == "s"


class TestSwapAliasRoundTrip:
    def test_swap_read_write_delete_and_schema(self):
        item = DataclassTwoFields(product_id="p", sku="s")
        aliases = {"product_id": "sku", "sku": "product_id"}
        adapter = ItemAdapter(item, field_aliases=aliases)

        assert adapter["sku"] == "p"
        assert adapter["product_id"] == "s"

        adapter["sku"] = "p2"
        adapter["product_id"] = "s2"
        assert item.product_id == "p2"
        assert item.sku == "s2"

        del adapter["sku"]
        assert "product_id" not in item.__dict__
        assert item.product_id == ""
        assert item.sku == "s2"

        schema = apply_aliases_to_json_schema(
            ItemAdapter.get_json_schema(DataclassTwoFields),
            ["product_id", "sku"],
            aliases,
            item_class_name=DataclassTwoFields.__name__,
        )
        assert list(schema["properties"]) == ["sku", "product_id"]

    def test_swap_rejected_when_only_one_field_is_renamed(self):
        with pytest.raises(ValueError, match="collides"):
            ItemAdapter(
                DataclassTwoFields(),
                field_aliases={"product_id": "sku"},
            )

    def test_swap_alias_collision_rejected_for_schema_export(self):
        from itemadapter._json_schema import apply_aliases_to_json_schema

        schema = ItemAdapter.get_json_schema(DataclassTwoFields)
        with pytest.raises(ValueError, match="collides"):
            apply_aliases_to_json_schema(
                schema,
                ["product_id", "sku"],
                {"product_id": "sku"},
                item_class_name=DataclassTwoFields.__name__,
            )


class TestSelfAlias:
    def test_alias_equal_to_own_field_name_is_allowed(self):
        @dataclass
        class SelfAliased:
            a: str = field(default="1", metadata={"alias": "a"})
            b: str = "2"

        adapter = ItemAdapter(SelfAliased())
        assert adapter["a"] == "1"
        assert sorted(adapter.field_names()) == ["a", "b"]

    def test_caller_self_alias_keeps_field_accessible(self):
        @dataclass
        class Plain:
            a: str = "1"
            b: str = "2"

        adapter = ItemAdapter(Plain(), field_aliases={"a": "a"})
        assert adapter["a"] == "1"
        assert dict(adapter) == {"a": "1", "b": "2"}
