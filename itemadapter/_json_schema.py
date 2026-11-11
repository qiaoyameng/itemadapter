from __future__ import annotations

import ast
import dataclasses
import inspect
import operator
import re
import sys
import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from copy import copy
from enum import Enum
from functools import lru_cache, reduce
from textwrap import dedent
from types import MappingProxyType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ForwardRef,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    runtime_checkable,
)

from ._imports import PydanticUndefined, _get_type_hints, _is_typeddict, attrs
from ._utils import _split_typed_dict_hint

if TYPE_CHECKING:
    from attrs import Attribute

    from .adapter import AdapterInterface, ItemAdapter


SIMPLE_TYPES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}


def _get_module_namespace(module_name: str) -> dict[str, Any]:
    """Return the global namespace of the module with the given name, or an
    empty dict if the module cannot be found."""
    module = sys.modules.get(module_name)
    if module is None:
        return {}
    return vars(module)


@dataclasses.dataclass
class _JsonSchemaState:
    adapter: type[ItemAdapter | AdapterInterface]
    """ItemAdapter class or AdapterInterface implementation used on the initial
    get_json_schema() call.

    On types for which adapter.is_item_class() returns True,
    adapter.get_json_schema() is used to get the corresponding, nested JSON
    Schema.
    """
    definitions: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    """$defs of the output JSON Schema, by definition name."""
    definition_identities: dict[Any, str] = dataclasses.field(default_factory=dict)
    """Definition name of each item class or generic alias that has been
    registered, so that the same identity always maps to the same $defs entry
    and $ref target.

    Generic aliases (e.g. Box[int]) are distinct identities from their origin
    class and from each other, so Box[int] and Box[str] get separate
    definitions.
    """
    processing: set[Any] = dataclasses.field(default_factory=set)
    """Identities whose definition is being built, to break recursion."""
    referenced: set[str] = dataclasses.field(default_factory=set)
    """$defs names that at least one $ref points at."""
    path: list[str] = dataclasses.field(default_factory=list)
    """Path of field names from the root item class to the field being
    processed, for error messages."""
    globalns: dict[str, Any] | None = None
    """Explicit global namespace for forward reference resolution, or None to
    use the namespace of the module where each class is defined."""
    localns: dict[str, Any] | None = None
    """Explicit local namespace for forward reference resolution."""

    def get_definition_name(self, identity: Any) -> str | None:
        """Return the $defs name registered for *identity*, or None."""
        return self.definition_identities.get(identity)

    def register_definition(self, identity: Any) -> str:
        """Allocate and remember a $defs name for *identity*, without
        filling the definition yet.

        The empty placeholder is inserted into the $defs mapping so that
        recursive references can target it before the definition is built.
        """
        name = _definition_name(identity, set(self.definitions))
        self.definition_identities[identity] = name
        self.definitions[name] = {}
        return name

    @staticmethod
    def ref(name: str) -> dict[str, Any]:
        """Return a $ref JSON Schema pointing at the given $defs name."""
        return {"$ref": f"#/$defs/{name}"}


def _eval_type(type_hint: Any, globalns: dict[str, Any], localns: dict[str, Any] | None) -> Any:
    """typing._eval_type() with a signature compatible with Python 3.11+."""
    return typing._eval_type(type_hint, globalns, localns, ())  # noqa: SLF001


def _forward_ref_name(annotation: Any) -> str:
    """Return a readable name for a forward reference annotation, whether it is
    a plain string or a ForwardRef object."""
    if isinstance(annotation, ForwardRef):
        return annotation.__forward_arg__
    return str(annotation)


def _type_identity(type_hint: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return the origin class and the type arguments of a type hint."""
    origin = get_origin(type_hint)
    if origin is not None and isinstance(origin, type):
        return origin, get_args(type_hint)
    return type_hint, ()


def _type_var_map(origin: type, args: tuple[Any, ...]) -> dict[TypeVar, Any]:
    """Return a mapping from the type variables of *origin* to *args*."""
    if not args:
        return {}
    parameters = getattr(origin, "__parameters__", ())
    return dict(zip(parameters, args))


def _substitute_type_vars(type_hint: Any, substitution: Mapping[TypeVar, Any]) -> Any:
    """Replace the type variables in *type_hint* according to *substitution*."""
    if not substitution:
        return type_hint
    if isinstance(type_hint, TypeVar):
        return substitution.get(type_hint, type_hint)
    origin = get_origin(type_hint)
    if origin is None:
        return type_hint
    args = get_args(type_hint)
    if not args:
        return type_hint
    if origin is Annotated:
        # Keep the metadata of Annotated hints untouched.
        return Annotated[
            (_substitute_type_vars(args[0], substitution), *args[1:])  # type: ignore[misc]
        ]
    new_args = tuple(_substitute_type_vars(arg, substitution) for arg in args)
    if origin is UnionType:
        return reduce(operator.or_, new_args)
    if origin is Union:
        return Union[new_args]  # noqa: UP007
    try:
        return type_hint.copy_with(new_args)
    except AttributeError:
        if len(new_args) == 1:
            return origin[new_args[0]]
        return origin[new_args]


def _definition_name(type_hint: Any, taken: set[str]) -> str:
    """Return a unique $defs name for *type_hint*, not present in *taken*.

    The name is based on the qualified name of the origin class, plus the
    type arguments for generic aliases. Same-named classes from different
    modules get numbered suffixes, so no definition is silently overridden.
    """
    origin, args = _type_identity(type_hint)
    if isinstance(origin, type):
        base = origin.__qualname__
    else:
        base = str(type_hint)
    if args:
        arg_names = ", ".join(_definition_arg_name(arg) for arg in args)
        base = f"{base}[{arg_names}]"
    name = base
    counter = 2
    while name in taken:
        name = f"{base}__{counter}"
        counter += 1
    return name


def _definition_arg_name(arg: Any) -> str:
    """Return a name fragment for a generic type argument."""
    origin = get_origin(arg)
    if origin is not None:
        sub_args = ", ".join(_definition_arg_name(sub) for sub in get_args(arg))
        origin_name = origin.__qualname__ if isinstance(origin, type) else str(origin)
        return f"{origin_name}[{sub_args}]"
    if isinstance(arg, type):
        return arg.__qualname__
    return str(arg)


def dedupe_types(types: Sequence[Any]) -> list[Any]:
    seen = set()
    result = []
    # JSON Schema's number type includes integers, but not the other way around.
    numeric_type = float if float in types else int
    for t in types:
        key = numeric_type if t in (int, float) else t
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def update_prop_from_union(prop: dict[str, Any], prop_type: Any, state: _JsonSchemaState) -> None:
    prop_types = dedupe_types(get_args(prop_type))
    simple_types = [v for k, v in SIMPLE_TYPES.items() if k in prop_types]
    complex_types = [t for t in prop_types if t not in SIMPLE_TYPES]
    if not complex_types:
        prop.setdefault("type", simple_types)
        return
    new_any_of: list[dict[str, Any]] = []
    any_of = prop.setdefault("anyOf", new_any_of)
    if any_of is not new_any_of:
        return
    if simple_types:
        any_of.append({"type": simple_types if len(simple_types) > 1 else simple_types[0]})
    for complex_type in complex_types:
        complex_prop: dict[str, Any] = {}
        update_prop_from_type(complex_prop, complex_type, state)
        any_of.append(complex_prop)


@runtime_checkable
class ArrayProtocol(Protocol):
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: object) -> bool: ...


@runtime_checkable
class ObjectProtocol(Protocol):  # noqa: PLW1641
    def __getitem__(self, key: str) -> Any: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: object) -> bool: ...
    def keys(self) -> Iterable[str]: ...
    def items(self) -> Iterable[tuple[str, Any]]: ...
    def values(self) -> Iterable[Any]: ...
    def get(self, key: str, default: Any = ...) -> Any: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


# Flags that change the meaning of a pattern in a way that cannot be
# expressed in JSON Schema, where patterns are always flagless. re.ASCII is
# not one of them: it brings Python semantics closer to JSON Schema ones.
INVALID_PATTERN_FLAGS = re.IGNORECASE | re.MULTILINE

INLINE_FLAGS = {
    "a": re.ASCII,
    "i": re.IGNORECASE,
    "L": re.LOCALE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "u": re.UNICODE,
    "x": re.VERBOSE,
}

# Python only allows global inline flags at the start of a pattern.
LEADING_INLINE_FLAGS = re.compile(r"\(\?([aiLmsux]+)\)")

# Whitespace that re.VERBOSE ignores.
VERBOSE_WHITESPACE = frozenset(" \t\n\r\v\f")

# Escape sequences that have the same meaning in Python and in ECMA-262. \b
# and \B are included even though Python matches word boundaries against
# Unicode word characters, while ECMA-262 only matches them against ASCII
# ones.
SHARED_ESCAPE_LETTERS = frozenset("bdDfnrstvwSW")

# Escape sequences with an ECMA-262 equivalent. ECMA-262 patterns are never
# multiline, so ^ and $ can only match at the start and at the end of the
# string, like \A and \Z do in Python.
CONVERTED_ESCAPES = {"A": "^", "Z": "$"}

# Escape sequences made of a letter and a fixed number of hexadecimal digits.
HEX_ESCAPE_SIZES = {"x": 2, "u": 4}

HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

DIGITS = frozenset("0123456789")

# Groups that have the same meaning in Python and in ECMA-262.
SHARED_GROUP_PREFIXES = ("(?:", "(?=", "(?!", "(?<=", "(?<!")


def scan_escape(pattern: str, index: int, *, in_class: bool) -> tuple[int, str] | None:
    """Return the ECMA-262 translation of the escape sequence that starts at
    *index*, along with the index right after it, or ``None`` if that sequence
    has no ECMA-262 equivalent."""
    char = pattern[index + 1 : index + 2]
    if not char:
        return None
    if char in HEX_ESCAPE_SIZES:
        end = index + 2 + HEX_ESCAPE_SIZES[char]
        digits = pattern[index + 2 : end]
        if len(digits) != HEX_ESCAPE_SIZES[char] or not HEX_DIGITS.issuperset(digits):
            return None
        return end, pattern[index:end]
    if char in SHARED_ESCAPE_LETTERS or (char == "B" and not in_class):
        return index + 2, pattern[index : index + 2]
    if char in CONVERTED_ESCAPES and not in_class:
        return index + 2, CONVERTED_ESCAPES[char]
    if char in DIGITS:
        # Backreferences have the same meaning, but \0 and multi-digit escapes
        # may be octal escapes instead, which the 2 dialects number
        # differently.
        if in_class or char == "0" or pattern[index + 2 : index + 3] in DIGITS:
            return None
        return index + 2, pattern[index : index + 2]
    if char.isalnum():
        return None
    return index + 2, pattern[index : index + 2]


def scan_class(pattern: str, index: int) -> tuple[int, str] | None:
    """Return the ECMA-262 translation of the character class that starts at
    *index*, along with the index right after it, or ``None`` if that class
    has no ECMA-262 equivalent."""
    start = index
    index += 1
    if pattern[index : index + 1] == "^":
        index += 1
    if pattern[index : index + 1] == "]":
        # Python reads a leading ] as a literal, ECMA-262 as the end of an
        # empty class.
        return None
    while index < len(pattern):
        char = pattern[index]
        if char == "]":
            return index + 1, pattern[start : index + 1]
        if char == "\\":
            # Escape sequences are never translated within a character class,
            # so the class can be copied verbatim.
            escape = scan_escape(pattern, index, in_class=True)
            if escape is None:
                return None
            index = escape[0]
            continue
        index += 1
    return None


def scan_group(pattern: str, index: int) -> tuple[int, str] | None:
    """Return the ECMA-262 translation of the opening parenthesis of the group
    that starts at *index*, along with the index right after it, or ``None``
    if that group has no ECMA-262 equivalent."""
    if pattern[index + 1 : index + 2] != "?":
        return index + 1, "("
    if pattern.startswith("(?#", index):
        end = pattern.find(")", index)
        return None if end == -1 else (end + 1, "")
    if pattern.startswith("(?P<", index):
        # ECMA-262 group names follow different rules, and JSON Schema never
        # reads captured groups, so the group is left unnamed. It stays a
        # capturing group to keep numbered backreferences working.
        end = pattern.find(">", index)
        return None if end == -1 else (end + 1, "(")
    for prefix in SHARED_GROUP_PREFIXES:
        if pattern.startswith(prefix, index):
            return index + len(prefix), prefix
    return None


def scan_braces(pattern: str, index: int, *, verbose: bool) -> tuple[int, str] | None:
    """Return the ECMA-262 translation of the quantifier or literal brace that
    starts at *index*, along with the index right after it, or ``None`` if it
    has no ECMA-262 equivalent."""
    end = pattern.find("}", index)
    if end == -1:
        return index + 1, "{"
    if pattern[end + 1 : end + 2] == "+":
        return None  # possessive quantifier
    if verbose and not VERBOSE_WHITESPACE.isdisjoint(pattern[index:end]):
        # Python drops the whitespace and reads the result as a literal,
        # ECMA-262 reads it as a quantifier.
        return None
    return end + 1, pattern[index : end + 1]


def translate_pattern(pattern: str, flags: int) -> str | None:
    """Return *pattern*, a Python regular expression with *flags*, as an
    ECMA-262 one, the regular expression dialect of JSON Schema patterns, or
    ``None`` if it has no ECMA-262 equivalent.

    See https://ecma-international.org/publications-and-standards/standards/ecma-262/
    """
    verbose = bool(flags & re.VERBOSE)
    output = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            result = scan_escape(pattern, index, in_class=False)
        elif char == "[":
            result = scan_class(pattern, index)
        elif char == "(":
            result = scan_group(pattern, index)
        elif char == "{":
            result = scan_braces(pattern, index, verbose=verbose)
        elif char in "*+?" and pattern[index + 1 : index + 2] == "+":
            return None  # possessive quantifier
        elif verbose and char in VERBOSE_WHITESPACE:
            result = (index + 1, "")
        elif verbose and char == "#":
            end = pattern.find("\n", index)
            result = (len(pattern) if end == -1 else end + 1, "")
        elif char == "." and flags & re.DOTALL:
            result = (index + 1, r"[\s\S]")
        else:
            result = (index + 1, char)
        if result is None:
            return None
        index, text = result
        output.append(text)
    return "".join(output)


def json_schema_pattern(pattern: str | re.Pattern[str]) -> str | None:
    """Return *pattern* as a JSON Schema pattern, or ``None`` if it cannot be
    expressed as one."""
    flags = 0
    if isinstance(pattern, re.Pattern):
        flags = pattern.flags
        pattern = pattern.pattern
    match = LEADING_INLINE_FLAGS.match(pattern)
    if match:
        for letter in match[1]:
            flags |= INLINE_FLAGS[letter]
        pattern = pattern[match.end() :]
    if flags & INVALID_PATTERN_FLAGS:
        return None
    return translate_pattern(pattern, flags)


def array_type(type_hint: Any) -> Any:
    """Given the type hint of a Python type that maps to a JSON Schema array,
    such as a list, a tuple or a set, return the type of the items in that
    array."""
    args = get_args(type_hint)
    if not args:
        return Any
    if args[-1] is Ellipsis:
        args = args[:-1]
    unique_args = set(args)
    if len(unique_args) == 1:
        return next(iter(unique_args))
    return Union[tuple(unique_args)]  # noqa: UP007


# JSON Schema patterns are unanchored, i.e. they behave like re.search(), so
# patterns meant for re.match() or re.fullmatch() need explicit anchors.
MATCH_FUNC_ANCHORS = {
    "search": ("", ""),
    "match": ("^(?:", ")"),
    "fullmatch": ("^(?:", ")$"),
}


def update_prop_from_pattern(prop: dict[str, Any], pattern: str | re.Pattern[str]) -> None:
    json_schema_value = json_schema_pattern(pattern)
    if json_schema_value is not None:
        prop.setdefault("pattern", json_schema_value)


UNION_TYPES = {Union, UnionType}


def update_prop_from_origin(
    prop: dict[str, Any], origin: Any, prop_type: Any, state: _JsonSchemaState
) -> None:
    if isinstance(origin, type):
        if issubclass(origin, (Sequence, AbstractSet)):
            prop.setdefault("type", "array")
            if issubclass(origin, AbstractSet):
                prop.setdefault("uniqueItems", True)
            had_items = "items" in prop
            items = prop.setdefault("items", {})
            item_type = array_type(prop_type)
            update_prop_from_type(items, item_type, state)
            if not items and not had_items:
                del prop["items"]
            return
        if issubclass(origin, Mapping):
            prop.setdefault("type", "object")
            args = get_args(prop_type)
            if args:
                assert len(args) == 2
                value_type = args[1]
                props = prop.setdefault("additionalProperties", {})
                update_prop_from_type(props, value_type, state)
            return
    if origin in UNION_TYPES:
        update_prop_from_union(prop, prop_type, state)


def _is_item_type_hint(prop_type: Any, state: _JsonSchemaState) -> bool:
    """Return True if *prop_type* is an item class, or a generic alias of one,
    that any registered adapter can handle.

    Detection always goes through ItemAdapter rather than the adapter of the
    containing item class, so that cross-type nesting, e.g. a TypedDict field
    of a dataclass, is recognized regardless of the containing type. Generic
    containers (lists, mappings, unions) are not item classes even when their
    origin is one of the handled types, e.g. dict.
    """
    from .adapter import ItemAdapter as _ItemAdapter

    if isinstance(prop_type, ForwardRef):
        return False
    if isinstance(prop_type, type):
        # The built-in dict is a valid item class for the adapter machinery,
        # but as a type hint it is a plain JSON Schema object, not a reusable
        # type to put in $defs.
        if prop_type is dict:
            return False
        try:
            return _ItemAdapter.is_item_class(prop_type)
        except TypeError:
            return False
    origin = get_origin(prop_type)
    if not isinstance(origin, type):
        return False
    # TypedDict classes are dict subclasses, hence Mapping subclasses, but
    # they are item classes that must be represented through $defs/$ref.
    if not _is_typeddict(origin) and (
        issubclass(origin, (Sequence, AbstractSet, Mapping)) or origin in UNION_TYPES
    ):
        return False
    try:
        return _ItemAdapter.is_item_class(origin)
    except TypeError:
        return False


def _update_prop_from_item(prop: dict[str, Any], prop_type: Any, state: _JsonSchemaState) -> bool:
    """Handle a type hint that maps to a reusable item class.

    The item class is represented through a $defs definition and a $ref. The
    definition is built once per identity (class plus generic parameters).

    Return True if the type hint was an item class and handled.
    """
    if not _is_item_type_hint(prop_type, state):
        return False
    from .adapter import ItemAdapter as _ItemAdapter

    identity = prop_type
    name = state.get_definition_name(identity)
    if name is None:
        name = state.register_definition(identity)
        state.processing.add(identity)
        try:
            # Dispatch through ItemAdapter so that cross-type nesting, e.g. a
            # TypedDict field of a dataclass, is built by the matching adapter.
            definition = _ItemAdapter.get_json_schema(prop_type, _state=state)
        finally:
            state.processing.discard(identity)
        state.definitions[name] = definition
    state.referenced.add(name)
    if not prop:
        prop.update(state.ref(name))
    else:
        prop.setdefault("$ref", f"#/$defs/{name}")
    return True


def update_prop_from_type(prop: dict[str, Any], prop_type: Any, state: _JsonSchemaState) -> None:
    if _update_prop_from_item(prop, prop_type, state):
        return
    if (origin := get_origin(prop_type)) is not None:
        update_prop_from_origin(prop, origin, prop_type, state)
        return
    if isinstance(prop_type, type):
        if issubclass(prop_type, Enum):
            values = [item.value for item in prop_type]
            value_types = tuple({type(v) for v in values})
            prop_type = value_types[0] if len(value_types) == 1 else Union[value_types]  # noqa: UP007
            update_prop_from_type(prop, prop_type, state)
            prop.setdefault("enum", values)
            return
        if not issubclass(prop_type, str):
            if isinstance(prop_type, ObjectProtocol):
                prop.setdefault("type", "object")
                return
            if isinstance(prop_type, ArrayProtocol):
                prop.setdefault("type", "array")
                if issubclass(prop_type, AbstractSet):
                    prop.setdefault("uniqueItems", True)
                return
    json_schema_type = SIMPLE_TYPES.get(prop_type)
    if json_schema_type is not None:
        prop.setdefault("type", json_schema_type)


def _setdefault_attribute_types_on_json_schema(
    schema: dict[str, Any], item_class: type, state: _JsonSchemaState
) -> None:
    """Inspect the type hints of the class attributes of the item class and,
    for any matching JSON Schema property that has no type set, set the type
    based on the type hint."""
    props = schema.get("properties", {})
    attribute_type_hints = _resolve_hints(item_class, state)
    for prop_name, prop in props.items():
        if prop_name not in attribute_type_hints:
            continue
        prop_type = attribute_type_hints[prop_name]
        state.path.append(prop_name)
        try:
            update_prop_from_type(prop, prop_type, state)
        finally:
            state.path.pop()


def iter_docstrings(item_class: type) -> Iterator[tuple[str, str]]:
    try:
        source = inspect.getsource(item_class)
    except (OSError, TypeError):
        return
    tree = ast.parse(dedent(source))
    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == item_class.__name__:
            class_node = node
            break
    if class_node is None:  # pragma: no cover
        # This can be reproduced with the doctests of the README, but the
        # coverage data does not seem to include those.
        return
    assert isinstance(class_node, ast.ClassDef)
    for child in ast.iter_child_nodes(class_node):
        if isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name):
            attr_name = child.targets[0].id
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            attr_name = child.target.id
        else:
            continue
        next_idx = class_node.body.index(child) + 1
        if next_idx >= len(class_node.body):
            continue
        next_node = class_node.body[next_idx]
        if (
            isinstance(next_node, ast.Expr)
            and isinstance(next_node.value, ast.Constant)
            and isinstance(next_node.value.value, str)
        ):
            yield attr_name, next_node.value.value


@lru_cache(maxsize=1024)
def get_attr_docstrings(item_class: type) -> Mapping[str, str]:
    """Return the docstrings of the attributes defined in the source of the
    given class, by attribute name."""
    docstrings: dict[str, str] = {}
    for attr_name, doc in iter_docstrings(item_class):
        docstrings.setdefault(attr_name, doc)
    return MappingProxyType(docstrings)


def get_inherited_attr_docstring(item_class: type, attr_name: str) -> str | None:
    """Recursively search the MRO for a docstring for the given attribute
    name."""
    for cls in item_class.__mro__:
        docstrings = get_attr_docstrings(cls)
        if attr_name in docstrings:
            return docstrings[attr_name]
    return None


def _setdefault_attribute_docstrings_on_json_schema(
    schema: dict[str, Any], item_class: type
) -> None:
    """Inspect the docstrings after each class attribute of the item class and
    its bases and, for any matching JSON Schema property that has no
    description set, set the description to the contents of the docstring."""
    props = schema.get("properties", {})
    attr_names = set(props)
    if not attr_names:
        return
    for attr_name in attr_names:
        prop = props.setdefault(attr_name, {})
        if "description" not in prop:
            doc = get_inherited_attr_docstring(item_class, attr_name)
            if doc:
                prop["description"] = inspect.cleandoc(doc)


def base_json_schema_from_item_class(item_class: type) -> dict[str, Any]:
    json_schema_extra = getattr(item_class, "__json_schema_extra__", {})
    schema = copy(json_schema_extra)
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    return schema


def _new_state(
    adapter: type[AdapterInterface],
    item_class: Any,
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
) -> _JsonSchemaState:
    """Return a fresh state for a root get_json_schema() call.

    The state holds the $defs registry for a single export, so no naming
    information is shared with previous or later exports.
    """
    origin, _ = _type_identity(item_class)
    root_class = origin if isinstance(origin, type) else item_class
    root_name = getattr(root_class, "__qualname__", str(root_class))
    return _JsonSchemaState(
        adapter=adapter,
        path=[root_name],
        globalns=globalns,
        localns=localns,
    )


def _finalize_root_schema(schema: dict[str, Any], state: _JsonSchemaState) -> dict[str, Any]:
    """Attach the collected $defs to the root schema, if any."""
    if state.definitions:
        schema["$defs"] = state.definitions
    return schema


def _json_schema_root(
    adapter: type[AdapterInterface],
    item_class: Any,
    builder: Any,
    globalns: dict[str, Any] | None = None,
    localns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the root JSON Schema of an item class or generic alias with a
    fresh state.

    The root item class is registered as a $defs definition before its schema
    is built, so that self references resolve through a $ref instead of
    recursing forever. The $defs registry lives in the state of this single
    export, so no naming information leaks into later exports.
    """
    state = _new_state(adapter, item_class, globalns, localns)
    name = state.register_definition(item_class)
    state.processing.add(item_class)
    try:
        schema = builder(item_class, state)
    finally:
        state.processing.discard(item_class)
    if name in state.referenced:
        # The root class is referenced (e.g. through a self reference), so its
        # definition must stay in $defs. Store a copy: the same schema is also
        # returned at the top level, and sharing the object would make the
        # output a circular reference.
        state.definitions[name] = copy(schema)
    else:
        # No $ref points at the root class, so its eagerly created definition
        # is removed to keep $defs limited to reusable types.
        del state.definitions[name]
    return _finalize_root_schema(schema, state)


def _json_schema_from_item_class(
    adapter: type[AdapterInterface], item_class: type, state: _JsonSchemaState | None = None
) -> dict[str, Any]:
    state = state or _new_state(adapter, item_class)
    schema = base_json_schema_from_item_class(item_class)
    fields_meta = {
        field_name: adapter.get_field_meta_from_class(item_class, field_name)
        for field_name in adapter.get_field_names_from_class(item_class) or ()
    }
    if not fields_meta:
        return schema
    schema["properties"] = {
        field_name: copy(field_meta.get("json_schema_extra", {}))
        for field_name, field_meta in fields_meta.items()
    }
    required = [
        field_name
        for field_name, field_data in schema["properties"].items()
        if "default" not in field_data
    ]
    if required:
        schema.setdefault("required", required)
    return schema


def update_required_fields(
    schema: dict[str, Any], optional_fields: set[str] | None = None
) -> None:
    optional_fields = optional_fields or set()
    if "required" in schema:
        return
    required = [
        field
        for field, metadata in schema["properties"].items()
        if field not in optional_fields and "default" not in metadata
    ]
    if required:
        schema["required"] = required


def _resolve_hints(
    item_class: type,
    state: _JsonSchemaState,
    *,
    include_extras: bool = False,
) -> dict[str, Any]:
    """Return the resolved type hints of *item_class*, using the namespaces
    of the state for forward reference resolution."""
    globalns = state.globalns
    if globalns is None:
        globalns = _get_module_namespace(item_class.__module__)
    localns = state.localns
    try:
        return _get_type_hints(
            item_class,
            globalns,
            localns,
            include_extras=include_extras,
        )
    except NameError as ex:
        # Identify the field whose forward reference cannot be resolved, so
        # that the error points at a field path instead of being swallowed.
        for field_name, annotation in getattr(item_class, "__annotations__", {}).items():
            ref = ForwardRef(annotation) if isinstance(annotation, str) else annotation
            try:
                _eval_type(ref, globalns, localns)
            except NameError:
                raise ValueError(
                    f"Cannot resolve forward reference {_forward_ref_name(annotation)!r} "
                    f"of field {'.'.join([*state.path, field_name])!r} of "
                    f"{item_class.__qualname__}: {ex}"
                ) from ex
        raise ValueError(
            f"Cannot resolve a forward reference of {item_class.__qualname__}: {ex}"
        ) from ex


def _json_schema_from_attrs(item_class: Any, state: _JsonSchemaState) -> dict[str, Any]:
    origin, args = _type_identity(item_class)
    substitution = _type_var_map(origin, args)
    schema = base_json_schema_from_item_class(origin)
    fields = attrs.fields(origin)
    if not fields:
        return schema

    globalns = state.globalns
    if globalns is None:
        globalns = _get_module_namespace(origin.__module__)
    try:
        attrs.resolve_types(origin, globalns, state.localns)  # Resolve field.type annotations
    except NameError as ex:
        for field in fields:
            if isinstance(field.type, str):
                try:
                    _eval_type(ForwardRef(field.type), globalns, state.localns)
                except NameError:
                    raise ValueError(
                        f"Cannot resolve forward reference {field.type!r} of field "
                        f"{'.'.join([*state.path, field.name])!r} of "
                        f"{origin.__qualname__}: {ex}"
                    ) from ex
        raise ValueError(
            f"Cannot resolve a forward reference of {origin.__qualname__}: {ex}"
        ) from ex

    schema["properties"] = {
        field.name: copy(field.metadata.get("json_schema_extra", {})) for field in fields
    }
    default_factory_fields: set[str] = set()
    for field in fields:
        prop = schema["properties"][field.name]
        state.path.append(field.name)
        try:
            _update_attrs_prop(prop, field, state, default_factory_fields, substitution)
        finally:
            state.path.pop()
    update_required_fields(schema, default_factory_fields)
    _setdefault_attribute_docstrings_on_json_schema(schema, origin)
    return schema


def _update_attrs_prop(
    prop: dict[str, Any],
    field: Attribute[Any],
    state: _JsonSchemaState,
    default_factory_fields: set[str],
    substitution: Mapping[TypeVar, Any] | None = None,
) -> None:
    field_type = _substitute_type_vars(field.type, substitution or {})
    update_prop_from_type(prop, field_type, state)
    if isinstance(field.default, attrs.Factory):
        # The factory is never called: only its presence affects requiredness.
        default_factory_fields.add(field.name)
    elif field.default is not attrs.NOTHING:
        prop.setdefault("default", field.default)
    _update_attrs_prop_validation(prop, field)


ATTRS_NUMBER_VALIDATORS = {
    operator.ge: "minimum",
    operator.gt: "exclusiveMinimum",
    operator.le: "maximum",
    operator.lt: "exclusiveMaximum",
}


def _update_attrs_prop_validation(
    prop: dict[str, Any],
    field: Attribute[Any],
) -> None:
    if not field.validator:
        return
    if type(field.validator).__name__ == "_AndValidator":
        validators = field.validator._validators  # type: ignore[attr-defined]
    else:
        validators = [field.validator]
    for validator in validators:
        validator_type_name = type(validator).__name__
        if validator_type_name == "_NumberValidator":
            key = ATTRS_NUMBER_VALIDATORS.get(validator.compare_func)
            if not key:  # pragma: no cover
                continue
            prop.setdefault(key, validator.bound)
        elif validator_type_name == "_InValidator":
            prop.setdefault("enum", list(validator.options))
        elif validator_type_name == "_MinLengthValidator":
            key = "minLength" if field.type is str else "minItems"
            prop.setdefault(key, validator.min_length)
        elif validator_type_name == "_MaxLengthValidator":
            key = "maxLength" if field.type is str else "maxItems"
            prop.setdefault(key, validator.max_length)
        elif validator_type_name == "_MatchesReValidator":
            pattern = json_schema_pattern(validator.pattern)
            if pattern is not None:
                prefix, suffix = MATCH_FUNC_ANCHORS[validator.match_func.__name__]
                prop.setdefault("pattern", f"{prefix}{pattern}{suffix}")


def _json_schema_from_typed_dict(item_class: Any, state: _JsonSchemaState) -> dict[str, Any]:
    origin, args = _type_identity(item_class)
    substitution = _type_var_map(origin, args)
    schema = base_json_schema_from_item_class(origin)
    type_hints = _resolve_hints(origin, state)
    if not type_hints:
        return schema
    extra_type_hints = _resolve_hints(origin, state, include_extras=True)
    schema["properties"] = {}
    required = []
    required_keys = origin.__required_keys__  # type: ignore[attr-defined]
    for field_name, field_type in type_hints.items():
        field_metadata, field_required = _split_typed_dict_hint(extra_type_hints[field_name])
        prop = copy(field_metadata.get("json_schema_extra", {}))
        field_type = _substitute_type_vars(field_type, substitution)
        state.path.append(field_name)
        try:
            update_prop_from_type(prop, field_type, state)
        finally:
            state.path.pop()
        schema["properties"][field_name] = prop
        if field_required is None:
            field_required = field_name in required_keys
        if field_required:
            required.append(field_name)
    if required:
        schema.setdefault("required", required)
    _setdefault_attribute_docstrings_on_json_schema(schema, origin)
    return schema


def _json_schema_from_dataclass(item_class: Any, state: _JsonSchemaState) -> dict[str, Any]:
    origin, args = _type_identity(item_class)
    substitution = _type_var_map(origin, args)
    schema = base_json_schema_from_item_class(origin)
    fields = dataclasses.fields(origin)
    resolved_field_types = _resolve_hints(origin, state)
    default_factory_fields = set()
    if fields:
        schema["properties"] = {
            field.name: copy(field.metadata.get("json_schema_extra", {})) for field in fields
        }
        for field in fields:
            prop = schema["properties"][field.name]
            field_type = _substitute_type_vars(resolved_field_types.get(field.name), substitution)
            state.path.append(field.name)
            try:
                update_prop_from_type(prop, field_type, state)
            finally:
                state.path.pop()
            if field.default_factory is not dataclasses.MISSING:
                # The factory is never called: only its presence affects
                # requiredness.
                default_factory_fields.add(field.name)
            elif field.default is not dataclasses.MISSING:
                prop.setdefault("default", field.default)
        update_required_fields(schema, default_factory_fields)
    _setdefault_attribute_docstrings_on_json_schema(schema, origin)
    return schema


def _json_schema_from_pydantic(
    adapter: type[AdapterInterface], item_class: type, state: _JsonSchemaState | None = None
) -> dict[str, Any]:
    state = state or _new_state(adapter, item_class)
    schema: dict[str, Any] = copy(
        item_class.model_config.get("json_schema_extra", {})  # type: ignore[attr-defined]
    )
    extra = item_class.model_config.get("extra")  # type: ignore[attr-defined]
    schema.setdefault("type", "object")
    if extra == "forbid":
        schema.setdefault("additionalProperties", False)
    fields = {
        name: adapter.get_field_meta_from_class(item_class, name)
        for name in adapter.get_field_names_from_class(item_class) or ()
    }
    if not fields:
        return schema
    schema["properties"] = {
        name: copy(metadata.get("json_schema_extra", {})) for name, metadata in fields.items()
    }
    default_factory_fields: set[str] = set()
    for name, metadata in fields.items():
        prop = schema["properties"][name]
        _update_pydantic_prop(prop, name, metadata, state, default_factory_fields)
    update_required_fields(schema, default_factory_fields)
    _setdefault_attribute_docstrings_on_json_schema(schema, item_class)
    return schema


def _update_pydantic_prop(
    prop: dict[str, Any],
    name: str,
    metadata: Mapping[str, Any],
    _state: _JsonSchemaState,
    default_factory_fields: set[str],
) -> None:
    field_type = metadata.get("annotation")
    update_prop_from_type(prop, field_type, _state)
    if "default_factory" in metadata:
        default_factory_fields.add(name)
    elif "default" in metadata and metadata["default"] is not PydanticUndefined:
        prop.setdefault("default", metadata["default"])
    if "metadata" in metadata:
        _update_pydantic_prop_validation(prop, metadata["metadata"], field_type)
    for metadata_key, json_schema_field in (
        ("title", "title"),
        ("description", "description"),
        ("examples", "examples"),
    ):
        if metadata_key in metadata:
            prop.setdefault(json_schema_field, metadata[metadata_key])
    if "deprecated" in metadata:
        prop.setdefault("deprecated", bool(metadata["deprecated"]))


def _update_pydantic_prop_validation(
    prop: dict[str, Any],
    metadata: Sequence[Any],
    field_type: object,
) -> None:
    for metadata_item in metadata:
        metadata_item_type = type(metadata_item).__name__
        if metadata_item_type == "_PydanticGeneralMetadata":
            if "pattern" in metadata_item.__dict__:
                pattern = metadata_item.__dict__["pattern"]
                update_prop_from_pattern(prop, pattern)
        elif metadata_item_type == "MinLen":
            key = "minLength" if field_type is str else "minItems"
            prop.setdefault(key, metadata_item.min_length)
        elif metadata_item_type == "MaxLen":
            key = "maxLength" if field_type is str else "maxItems"
            prop.setdefault(key, metadata_item.max_length)
        else:
            for metadata_key, json_schema_field in (
                ("ge", "minimum"),
                ("gt", "exclusiveMinimum"),
                ("le", "maximum"),
                ("lt", "exclusiveMaximum"),
            ):
                if metadata_item_type == metadata_key.capitalize():
                    prop.setdefault(json_schema_field, getattr(metadata_item, metadata_key))
