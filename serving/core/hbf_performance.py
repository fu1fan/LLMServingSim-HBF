"""Operator-latency sources for weights resident in HBF."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType

from .hbf_model import is_hbf_location


def apply_hbf_latency_scale_override(instances, latency_scale):
    if latency_scale is None:
        return
    if latency_scale <= 0:
        raise ValueError("--hbf-latency-scale must be positive")

    hbf_instances = [
        instance for instance in instances if instance.get("hbf_mem")
    ]
    if not hbf_instances:
        raise ValueError(
            "--hbf-latency-scale requires at least one hbf_mem instance"
        )
    for instance in hbf_instances:
        performance = instance["hbf_mem"]["performance"]
        if performance["source"] != "scale":
            raise ValueError(
                "--hbf-latency-scale cannot override an HBF profile source"
            )
        performance["latency_scale"] = float(latency_scale)


@dataclass(frozen=True)
class HBFOperatorQuery:
    model: str
    variant: str
    tp_size: int
    ep_size: int
    layer_name: str
    category: str
    shape: dict
    baseline_latency_ns: int
    weight_bytes: int
    weight_location: str

    def __post_init__(self):
        if self.baseline_latency_ns < 0:
            raise ValueError("baseline_latency_ns must be non-negative")
        if self.weight_bytes < 0:
            raise ValueError("weight_bytes must be non-negative")
        object.__setattr__(
            self,
            "shape",
            MappingProxyType(dict(self.shape)),
        )

    @property
    def shape_key(self):
        return self.shape

    @property
    def uses_hbf_weights(self):
        return self.weight_bytes > 0 and is_hbf_location(
            self.weight_location
        )


class HBFPerformanceSource(ABC):
    """Resolve the complete operator latency for one HBF-backed query."""

    source_name = "abstract"
    evidence_level = "unknown"

    @abstractmethod
    def latency_ns(self, query):
        raise NotImplementedError

    def describe(self):
        return {
            "source": self.source_name,
            "evidence_level": self.evidence_level,
        }


class IdentityHBFPerformanceSource(HBFPerformanceSource):
    """Identity source used to validate source wiring without changing time."""

    source_name = "identity"
    evidence_level = "identity"

    def latency_ns(self, query):
        return query.baseline_latency_ns


class ScaleHBFPerformanceSource(HBFPerformanceSource):
    source_name = "scale"
    evidence_level = "sensitivity-analysis"

    def __init__(self, latency_scale):
        if (
            not isinstance(latency_scale, (int, float))
            or isinstance(latency_scale, bool)
            or latency_scale <= 0
        ):
            raise ValueError("HBF latency_scale must be a positive number")
        self.latency_scale = float(latency_scale)

    def latency_ns(self, query):
        return int(round(query.baseline_latency_ns * self.latency_scale))

    def describe(self):
        value = super().describe()
        value["latency_scale"] = self.latency_scale
        return value
