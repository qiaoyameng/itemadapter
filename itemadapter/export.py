from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from itemadapter.adapter import ItemAdapter

__all__ = [
    "BaseExportStrategy",
    "ExportDecision",
    "ExportError",
    "ExportPath",
    "FieldSegment",
    "FieldContext",
    "IndexSegment",
    "RuleExportStrategy",
    "export",
]


class ExportDecision(Enum):
    """What an :class:`ExportStrategy` decides to do with a field."""

    KEEP = "keep"
    EXCLUDE = "exclude"
    MASK = "mask"


@dataclass(frozen=True)
class FieldSegment:
    """A path segment that addresses an item field or a dictionary key."""

    name: str

    def render(self) -> str:
        return self.name


@dataclass(frozen=True)
class IndexSegment:
    """A path segment that addresses an element of a sequence."""

    index: int

    def render(self) -> str:
        return f"[{self.index}]"


Segment = FieldSegment | IndexSegment


@dataclass(frozen=True)
class ExportPath:
    """The structured location of a field within the exported object.

    A path is a sequence of :class:`FieldSegment` and :class:`IndexSegment`
    objects. It is shared by nested items, dictionaries and sequences reached
    while walking the same object tree.
    """

    segments: tuple[Segment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def child(self, segment: Segment) -> ExportPath:
        return ExportPath((*self.segments, segment))

    def as_tuple(self) -> tuple[str | int, ...]:
        """Return the path as a plain tuple of field names and indices."""
        return tuple(
            segment.name if isinstance(segment, FieldSegment) else segment.index
            for segment in self.segments
        )

    def render(self) -> str:
        if not self.segments:
            return "<root>"
        text = self.segments[0].render()
        for segment in self.segments[1:]:
            if isinstance(segment, IndexSegment):
                text += segment.render()
            else:
                text += f".{segment.render()}"
        return text

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class FieldContext:
    """Everything an :class:`ExportStrategy` can inspect before a value is read."""

    path: ExportPath
    field_name: str
    metadata: MappingProxyType[str, Any]
    item_class: type


def _element_context(path: ExportPath, index: int) -> FieldContext:
    """Synthetic context for a sequence element.

    Sequence elements have no field name, class or metadata, but they still
    participate in the shared path context, so path-based rules can keep,
    exclude or mask them.
    """
    element_path = path.child(IndexSegment(index))
    return FieldContext(
        path=element_path,
        field_name=str(index),
        metadata=MappingProxyType({}),
        item_class=object,
    )


class ExportError(Exception):
    """Raised when an object cannot be exported.

    Error messages identify the offending :class:`ExportPath` and the object
    type. They never include the offending value, so sensitive data cannot leak
    through exceptions raised during an export.
    """

    def __init__(self, message: str, path: ExportPath) -> None:
        self.path = path
        super().__init__(f"{message} (at {path.render()})")


@runtime_checkable
class ExportStrategy(Protocol):
    """Strategy deciding which fields an export keeps, excludes or masks.

    The decision is made from field metadata and the structured field path,
    before the field value is ever read.
    """

    def decide(self, context: FieldContext) -> ExportDecision: ...

    def redact(self, context: FieldContext) -> Any: ...


class BaseExportStrategy:
    """Convenience base for custom export strategies.

    Subclasses must implement :meth:`decide` and :meth:`redact`. Sequence
    elements have no field metadata, so they are kept by default; override
    :meth:`decide_element` to exclude or mask elements by their path.
    """

    @abstractmethod
    def decide(self, context: FieldContext) -> ExportDecision:
        raise NotImplementedError

    @abstractmethod
    def redact(self, context: FieldContext) -> Any:
        raise NotImplementedError

    def decide_element(self, path: ExportPath, index: int) -> ExportDecision:
        """Keep every sequence element unless overridden."""
        return ExportDecision.KEEP


def _normalize_path(path: ExportPath | Sequence[str | int | Segment]) -> ExportPath:
    if isinstance(path, ExportPath):
        return path
    segments: list[Segment] = []
    for part in path:
        if isinstance(part, (FieldSegment, IndexSegment)):
            segments.append(part)
        elif isinstance(part, str):
            segments.append(FieldSegment(part))
        elif isinstance(part, int) and not isinstance(part, bool):
            segments.append(IndexSegment(part))
        else:
            raise TypeError(
                f"Path parts must be str, int or segment objects, got {type(part).__name__}"
            )
    return ExportPath(tuple(segments))


def _priority(decision: ExportDecision) -> int:
    # Exclude always wins over mask, which always wins over keep.
    return {
        ExportDecision.KEEP: 0,
        ExportDecision.MASK: 1,
        ExportDecision.EXCLUDE: 2,
    }[decision]


class RuleExportStrategy(BaseExportStrategy):
    """An :class:`ExportStrategy` driven by explicit path and metadata rules.

    Path rules match the full :class:`ExportPath` of each field. Metadata rules
    match fields whose metadata maps ``metadata_key`` to one of the given
    values. Both kinds of rules are combined with a fixed priority: exclusion
    beats masking, which beats keeping. A field without any matching rule is
    kept.
    """

    def __init__(
        self,
        *,
        exclude_paths: Iterable[ExportPath | Sequence[str | int | Segment]] = (),
        include_paths: Iterable[ExportPath | Sequence[str | int | Segment]] = (),
        mask_paths: Iterable[ExportPath | Sequence[str | int | Segment]] = (),
        exclude_metadata: Iterable[Any] = (),
        include_metadata: Iterable[Any] = (),
        mask_metadata: Iterable[Any] = (),
        metadata_key: str = "export",
        redact_value: Any = "<redacted>",
    ) -> None:
        self._exclude_paths = {
            _normalize_path(path).as_tuple() for path in exclude_paths
        }
        self._include_paths = {
            _normalize_path(path).as_tuple() for path in include_paths
        }
        self._mask_paths = {_normalize_path(path).as_tuple() for path in mask_paths}
        self._exclude_metadata = set(exclude_metadata)
        self._include_metadata = set(include_metadata)
        self._mask_metadata = set(mask_metadata)
        self._metadata_key = metadata_key
        self._redact_value = redact_value

    def decide(self, context: FieldContext) -> ExportDecision:
        path_tuple = context.path.as_tuple()
        decision = ExportDecision.KEEP
        if path_tuple in self._exclude_paths:
            decision = ExportDecision.EXCLUDE
        if path_tuple in self._mask_paths:
            decision = self._stronger(decision, ExportDecision.MASK)
        if path_tuple in self._include_paths:
            decision = self._stronger(decision, ExportDecision.KEEP)
        try:
            metadata_value = context.metadata.get(self._metadata_key)
        except AttributeError:
            metadata_value = None
        if metadata_value in self._exclude_metadata:
            decision = self._stronger(decision, ExportDecision.EXCLUDE)
        if metadata_value in self._mask_metadata:
            decision = self._stronger(decision, ExportDecision.MASK)
        if metadata_value in self._include_metadata:
            decision = self._stronger(decision, ExportDecision.KEEP)
        return decision

    @staticmethod
    def _stronger(
        current: ExportDecision, candidate: ExportDecision
    ) -> ExportDecision:
        return candidate if _priority(candidate) > _priority(current) else current

    def redact(self, context: FieldContext) -> Any:
        return self._redact_value

    def decide_element(self, path: ExportPath, index: int) -> ExportDecision:
        element_path = path.child(IndexSegment(index)).as_tuple()
        decision = ExportDecision.KEEP
        if element_path in self._exclude_paths:
            decision = ExportDecision.EXCLUDE
        if element_path in self._mask_paths:
            decision = self._stronger(decision, ExportDecision.MASK)
        if element_path in self._include_paths:
            decision = self._stronger(decision, ExportDecision.KEEP)
        return decision


_LEAF_TYPES = (type(None), bool, int, float, str, bytes)
_SEQUENCE_TYPES = (list, tuple, set, frozenset)


def export(
    obj: object,
    strategy: ExportStrategy,
    *,
    _path: ExportPath | None = None,
    _seen: set[int] | None = None,
) -> object:
    """Export ``obj`` into plain containers following ``strategy``.

    Excluded fields are decided on before their values are read, so their
    getters are never invoked. Masked fields are replaced through the
    strategy's ``redact`` callback, also without reading the original value.
    Nested items, dictionaries and sequences share the same path context, and
    unsupported types or reference cycles raise :class:`ExportError`.
    """
    from itemadapter.adapter import ItemAdapter

    path = _path or ExportPath()
    seen = _seen if _seen is not None else set()

    if isinstance(obj, ItemAdapter):
        return _export_item(obj, strategy, path, seen)
    if ItemAdapter.is_item(obj):
        return _export_item(ItemAdapter(obj), strategy, path, seen)
    if isinstance(obj, dict):
        return _export_dict(obj, strategy, path, seen)
    if isinstance(obj, _SEQUENCE_TYPES):
        return _export_sequence(obj, strategy, path, seen)
    if isinstance(obj, _LEAF_TYPES):
        return obj
    raise ExportError(
        f"Objects of type {type(obj).__name__} cannot be exported", path
    )


def _export_item(
    adapter: ItemAdapter,
    strategy: ExportStrategy,
    path: ExportPath,
    seen: set[int],
) -> dict[str, Any]:
    obj_id = id(adapter.item)
    if obj_id in seen:
        raise ExportError("Circular reference detected while exporting", path)
    seen.add(obj_id)
    try:
        result: dict[str, Any] = {}
        for field_name in adapter.field_names():
            if not isinstance(field_name, str):
                raise ExportError(
                    f"Dictionary keys of type {type(field_name).__name__} cannot be exported",
                    path,
                )
            field_path = path.child(FieldSegment(field_name))
            context = FieldContext(
                path=field_path,
                field_name=field_name,
                metadata=adapter.get_field_meta(field_name),
                item_class=adapter.item.__class__,
            )
            decision = strategy.decide(context)
            if decision is ExportDecision.EXCLUDE:
                continue
            if decision is ExportDecision.MASK:
                result[field_name] = strategy.redact(context)
                continue
            value = adapter[field_name]
            result[field_name] = export(
                value, strategy, _path=field_path, _seen=seen
            )
        return result
    finally:
        seen.discard(obj_id)


def _export_dict(
    obj: dict[Any, Any],
    strategy: ExportStrategy,
    path: ExportPath,
    seen: set[int],
) -> dict[Any, Any]:
    obj_id = id(obj)
    if obj_id in seen:
        raise ExportError("Circular reference detected while exporting", path)
    seen.add(obj_id)
    try:
        result: dict[Any, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ExportError(
                    f"Dictionary keys of type {type(key).__name__} cannot be exported",
                    path,
                )
            result[key] = export(
                value, strategy, _path=path.child(FieldSegment(key)), _seen=seen
            )
        return result
    finally:
        seen.discard(obj_id)


def _export_sequence(
    obj: Sequence[Any] | set[Any] | frozenset[Any],
    strategy: ExportStrategy,
    path: ExportPath,
    seen: set[int],
) -> Any:
    obj_id = id(obj)
    if obj_id in seen:
        raise ExportError("Circular reference detected while exporting", path)
    seen.add(obj_id)
    decide_element = getattr(strategy, "decide_element", None)

    def build(index: int, element: Any) -> Any:
        element_path = path.child(IndexSegment(index))
        if decide_element is not None:
            decision = decide_element(path, index)
            if decision is ExportDecision.EXCLUDE:
                return _OMITTED
            if decision is ExportDecision.MASK:
                return strategy.redact(_element_context(path, index))
        return export(element, strategy, _path=element_path, _seen=seen)

    try:
        if isinstance(obj, set):
            return {
                built
                for index, element in enumerate(obj)
                if (built := build(index, element)) is not _OMITTED
            }
        if isinstance(obj, frozenset):
            return frozenset(
                built
                for index, element in enumerate(obj)
                if (built := build(index, element)) is not _OMITTED
            )
        return obj.__class__(
            built
            for index, element in enumerate(obj)
            if (built := build(index, element)) is not _OMITTED
        )
    finally:
        seen.discard(obj_id)


class _Omitted:
    """Sentinel marking a sequence element dropped by the strategy."""


_OMITTED = _Omitted()
