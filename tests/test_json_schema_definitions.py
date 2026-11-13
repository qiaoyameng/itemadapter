from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from typing import Generic, TypeVar, TypedDict

import pytest

from itemadapter import ItemAdapter
from itemadapter._json_schema import JsonSchemaError

T = TypeVar("T")


@dataclass
class MutualA:
    b: MutualB


@dataclass
class MutualB:
    a: MutualA


@dataclass
class Box(Generic[T]):
    value: T


@dataclass
class BoxPair:
    int_box: Box[int]
    str_box: Box[str]


@dataclass
class ImpossibleFactoryItem:
    count: int = field(
        default=3,
        metadata={"json_schema_extra": {"minimum": 1}},
    )
    """The count."""

    values: list[int] = field(
        default_factory=lambda: (_ for _ in ()).throw(AssertionError("factory called"))
    )


@dataclass
class ImpossibleFactoryHolder:
    nested: ImpossibleFactoryItem


class NestedTypedDict(TypedDict):
    value: int


class TypedDictHolder(TypedDict):
    nested: NestedTypedDict


@dataclass
class BrokenForwardReference:
    child: Missing  # type: ignore[name-defined]  # noqa: F821


def iter_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref":
                yield item
            else:
                yield from iter_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_refs(item)


def assert_refs_resolve(schema):
    for ref in iter_refs(schema):
        assert ref.startswith("#/$defs/")
        name = ref[len("#/$defs/") :].replace("~1", "/").replace("~0", "~")
        assert name in schema["$defs"]


def test_mutually_recursive_definitions():
    schema = ItemAdapter.get_json_schema(MutualA)

    assert len(schema["$defs"]) == 2
    assert_refs_resolve(schema)
    assert schema["properties"]["b"]["$ref"].rsplit("/", 1)[-1].endswith("MutualB")
    assert schema["$defs"]["tests.test_json_schema_definitions.MutualA"]["properties"]["b"] == {
        "$ref": "#/$defs/tests.test_json_schema_definitions.MutualB"
    }


def test_generic_definitions_are_distinct():
    schema = ItemAdapter.get_json_schema(BoxPair)

    assert len(schema["$defs"]) == 2
    assert_refs_resolve(schema)
    assert schema["properties"]["int_box"]["$ref"] != schema["properties"]["str_box"]["$ref"]
    int_definition = next(
        definition
        for name, definition in schema["$defs"].items()
        if name.endswith("Box_int")
    )
    str_definition = next(
        definition
        for name, definition in schema["$defs"].items()
        if name.endswith("Box_str")
    )
    assert int_definition["properties"]["value"] == {"type": "integer"}
    assert str_definition["properties"]["value"] == {"type": "string"}


def test_defaults_metadata_and_descriptions_are_preserved():
    schema = ItemAdapter.get_json_schema(ImpossibleFactoryHolder)
    nested = next(iter(schema["$defs"].values()))

    assert nested["properties"]["count"] == {
        "minimum": 1,
        "type": "integer",
        "default": 3,
        "description": "The count.",
    }
    assert "default" not in nested["properties"]["values"]
    assert "required" not in nested


def test_typed_dict_definitions():
    schema = ItemAdapter.get_json_schema(TypedDictHolder)

    assert len(schema["$defs"]) == 1
    assert_refs_resolve(schema)
    definition = schema["$defs"]["tests.test_json_schema_definitions.NestedTypedDict"]
    assert definition["properties"]["value"] == {"type": "integer"}
    assert definition["required"] == ["value"]


def test_attrs_and_typed_dict_generic_definitions():
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class AttrsBox(Generic[T]):
        value: T

    class GenericTypedDict(TypedDict, Generic[T]):
        value: T

    @dataclass
    class Holder:
        attrs_int: AttrsBox[int]
        attrs_str: AttrsBox[str]
        typed_int: GenericTypedDict[int]
        typed_str: GenericTypedDict[str]

    schema = ItemAdapter.get_json_schema(
        Holder,
        namespace={"AttrsBox": AttrsBox, "GenericTypedDict": GenericTypedDict},
    )

    assert len(schema["$defs"]) == 4
    assert_refs_resolve(schema)
    values = sorted(
        definition["properties"]["value"]["type"]
        for definition in schema["$defs"].values()
    )
    assert values == ["integer", "integer", "string", "string"]


def test_same_named_classes_in_different_modules():
    module_a = types.ModuleType("definition_test_a")
    module_b = types.ModuleType("definition_test_b")
    sys.modules[module_a.__name__] = module_a
    sys.modules[module_b.__name__] = module_b
    try:
        exec(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Thing:\n"
            "    a: int",
            module_a.__dict__,
        )
        exec(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Thing:\n"
            "    b: str",
            module_b.__dict__,
        )

        @dataclass
        class Both:
            a: module_a.Thing
            b: module_b.Thing

        schema = ItemAdapter.get_json_schema(
            Both,
            namespace={"module_a": module_a, "module_b": module_b},
        )
    finally:
        sys.modules.pop(module_a.__name__, None)
        sys.modules.pop(module_b.__name__, None)

    assert len(schema["$defs"]) == 2
    assert "definition_test_a.Thing" in schema["$defs"]
    assert "definition_test_b.Thing" in schema["$defs"]
    assert_refs_resolve(schema)


def test_forward_references_use_the_given_namespace():
    def define_models():
        @dataclass
        class First:
            second: "Second"

        @dataclass
        class Second:
            first: "First"

        return First, Second

    first, second = define_models()
    schema = ItemAdapter.get_json_schema(first, namespace={"First": first, "Second": second})

    assert len(schema["$defs"]) == 2
    assert_refs_resolve(schema)


def test_unresolvable_forward_reference_reports_the_field_path():
    with pytest.raises(JsonSchemaError, match=r"BrokenForwardReference\.child"):
        ItemAdapter.get_json_schema(BrokenForwardReference)


def test_repeated_exports_are_stable_and_do_not_share_names():
    first = ItemAdapter.get_json_schema(BoxPair)
    second = ItemAdapter.get_json_schema(BoxPair)
    mutual = ItemAdapter.get_json_schema(MutualA)

    assert json.dumps(first) == json.dumps(second)
    assert len(mutual["$defs"]) == 2
    assert set(mutual["$defs"]).isdisjoint(first["$defs"])
    assert_refs_resolve(mutual)


def test_schemas_pass_draft_2020_12_validation():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator

    for item_class in (MutualA, BoxPair, TypedDictHolder):
        validator.check_schema(ItemAdapter.get_json_schema(item_class))
