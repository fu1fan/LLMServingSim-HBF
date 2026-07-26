"""Capacity accounting for High-Bandwidth Flash attached to one GPU."""

from dataclasses import dataclass, field
from enum import Enum


GB_TO_BYTE = 1024 * 1024 * 1024


class HBFAllocationKind(str, Enum):
    WEIGHT = "weight"
    KV = "kv"
    PREFIX = "prefix"


@dataclass
class HBFMemory:
    """Track HBF capacity by allocation class.

    Only static weights are wired into the serving runtime in the first
    implementation. KV and prefix accounting are intentionally part of this
    low-level interface so later offload policies do not need to replace the
    capacity model.
    """

    num_stacks: int
    stack_capacity_gb: float
    _used: dict = field(
        default_factory=lambda: {
            kind: 0 for kind in HBFAllocationKind
        },
        init=False,
        repr=False,
    )

    def __post_init__(self):
        if not isinstance(self.num_stacks, int) or self.num_stacks <= 0:
            raise ValueError("HBF num_stacks must be a positive integer")
        if (
            not isinstance(self.stack_capacity_gb, (int, float))
            or isinstance(self.stack_capacity_gb, bool)
            or self.stack_capacity_gb <= 0
        ):
            raise ValueError(
                "HBF stack_capacity_gb must be a positive number"
            )

    @property
    def capacity_bytes(self):
        return int(
            self.num_stacks * self.stack_capacity_gb * GB_TO_BYTE
        )

    @property
    def used_bytes(self):
        return sum(self._used.values())

    @property
    def available_bytes(self):
        return self.capacity_bytes - self.used_bytes

    def used_by_kind(self, kind):
        return self._used[self._coerce_kind(kind)]

    def allocate(self, size, kind):
        kind = self._coerce_kind(kind)
        size = self._validate_size(size)
        if size > self.available_bytes:
            raise RuntimeError(
                "HBF capacity exceeded: "
                f"required={size} available={self.available_bytes}"
            )
        self._used[kind] += size

    def free(self, size, kind):
        kind = self._coerce_kind(kind)
        size = self._validate_size(size)
        if size > self._used[kind]:
            raise RuntimeError(
                f"HBF {kind.value} underflow: "
                f"requested={size} used={self._used[kind]}"
            )
        self._used[kind] -= size

    def snapshot(self):
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "available_bytes": self.available_bytes,
            **{
                f"{kind.value}_used_bytes": self._used[kind]
                for kind in HBFAllocationKind
            },
        }

    @staticmethod
    def _coerce_kind(kind):
        try:
            return HBFAllocationKind(kind)
        except ValueError as exc:
            valid = ", ".join(item.value for item in HBFAllocationKind)
            raise ValueError(
                f"Unknown HBF allocation kind {kind!r}; expected {valid}"
            ) from exc

    @staticmethod
    def _validate_size(size):
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("HBF allocation size must be a non-negative int")
        return size
