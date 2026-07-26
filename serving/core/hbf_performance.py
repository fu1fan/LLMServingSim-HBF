"""Operator-latency sources for weights resident in HBF."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType

from .hbf_model import is_hbf_location


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
