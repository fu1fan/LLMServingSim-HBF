"""Configuration and capacity accounting for HBF attached to one GPU."""

from dataclasses import dataclass, field
from enum import Enum


GB_TO_BYTE = 1024 * 1024 * 1024


class HBFAllocationKind(str, Enum):
    WEIGHT = "weight"
    KV = "kv"
    PREFIX = "prefix"


@dataclass(frozen=True)
class HBFConfig:
    schema_version: int
    num_stacks: int
    stack_capacity_gb: float
    performance: dict

    @property
    def capacity_bytes(self):
        return int(
            self.num_stacks * self.stack_capacity_gb * GB_TO_BYTE
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "num_stacks": self.num_stacks,
            "stack_capacity_gb": self.stack_capacity_gb,
            "mem_size": self.num_stacks * self.stack_capacity_gb,
            "performance": dict(self.performance),
        }


def parse_hbf_config(value):
    """Validate an optional per-instance ``hbf_mem`` configuration."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("hbf_mem must be an object")

    required = {
        "schema_version",
        "num_stacks",
        "stack_capacity_gb",
        "performance",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise KeyError(
            f"Missing required hbf_mem keys: {', '.join(missing)}"
        )

    schema_version = value["schema_version"]
    if schema_version != 1:
        raise ValueError(
            f"Unsupported hbf_mem schema_version {schema_version!r}; expected 1"
        )

    # Reuse the capacity model's validation so the public configuration and
    # runtime allocator cannot disagree about valid stack dimensions.
    memory = HBFMemory(
        num_stacks=value["num_stacks"],
        stack_capacity_gb=value["stack_capacity_gb"],
    )

    performance = value["performance"]
    if not isinstance(performance, dict):
        raise TypeError("hbf_mem.performance must be an object")
    source = performance.get("source")
    if source not in {"scale", "profile"}:
        raise ValueError(
            "hbf_mem.performance.source must be 'scale' or 'profile'"
        )
    if source == "scale":
        latency_scale = performance.get("latency_scale")
        if (
            not isinstance(latency_scale, (int, float))
            or isinstance(latency_scale, bool)
            or latency_scale <= 0
        ):
            raise ValueError(
                "hbf_mem.performance.latency_scale must be positive"
            )
    else:
        for key in ("profile_root", "profile_hardware"):
            if not isinstance(performance.get(key), str) or not performance[key]:
                raise ValueError(
                    f"hbf_mem.performance.{key} must be a non-empty string"
                )

    return HBFConfig(
        schema_version=schema_version,
        num_stacks=memory.num_stacks,
        stack_capacity_gb=memory.stack_capacity_gb,
        performance=dict(performance),
    )


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
