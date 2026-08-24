"""Operator-latency sources for weights resident in HBF."""

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType

from .hbf_model import is_hbf_location


def apply_hbf_latency_scale_override(instances, latency_scale):
    if latency_scale is None:
        return
    if (
        not isinstance(latency_scale, (int, float))
        or isinstance(latency_scale, bool)
        or not math.isfinite(latency_scale)
        or latency_scale <= 0
    ):
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
            or not math.isfinite(latency_scale)
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


def build_hbf_performance_source(
    hbf_mem,
    model,
    variant,
    tp_needed,
    model_type,
):
    """Construct the configured source, or ``None`` for legacy instances."""
    if hbf_mem is None:
        return None
    performance = hbf_mem["performance"]
    source = performance["source"]
    if source == "scale":
        return ScaleHBFPerformanceSource(performance["latency_scale"])
    if source == "profile":
        return ProfileBundleHBFPerformanceSource(
            performance["profile_root"],
            performance["profile_hardware"],
            model,
            variant,
            tp_needed,
            model_type,
        )
    raise ValueError(f"Unsupported HBF performance source {source!r}")


class ProfileBundleHBFPerformanceSource(HBFPerformanceSource):
    source_name = "profile"
    evidence_level = "external-simulator-backed"

    def __init__(
        self,
        profile_root,
        profile_hardware,
        model,
        variant,
        tp_needed,
        model_type,
    ):
        # Imported lazily to keep the provider interface independent from the
        # trace generator while reusing its canonical lookup implementation.
        from . import trace_generator

        variant_root = os.path.join(
            os.path.abspath(profile_root),
            profile_hardware,
            model,
            variant,
        )
        self.perf_db = trace_generator._load_perf_db_at_variant_root(
            variant_root,
            profile_hardware,
            model,
            variant,
            set(tp_needed),
            model_type,
        )
        self._trace_generator = trace_generator
        self._validate_provenance(model, variant, profile_hardware)

    def _validate_provenance(self, model, variant, profile_hardware):
        meta = self.perf_db.get("meta") or {}
        hbf_meta = meta.get("hbf_profile")
        if not isinstance(hbf_meta, dict):
            raise ValueError(
                "External HBF profile meta.yaml must contain hbf_profile"
            )
        if hbf_meta.get("schema_version") != 1:
            raise ValueError(
                "External HBF profile requires hbf_profile.schema_version=1"
            )
        for key in ("producer", "source"):
            if not isinstance(hbf_meta.get(key), str) or not hbf_meta[key]:
                raise ValueError(
                    f"External HBF profile requires hbf_profile.{key}"
                )
        expected = {
            "model": model,
            "variant": variant,
            "hardware": profile_hardware,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"External HBF profile {key} mismatch: "
                    f"expected={value!r} actual={meta.get(key)!r}"
                )
        self.provenance = dict(hbf_meta)

    def latency_ns(self, query):
        tg = self._trace_generator
        shape = query.shape_key
        if query.category == "dense":
            return tg._lookup_dense(
                self.perf_db,
                query.layer_name,
                query.tp_size,
                shape["tokens"],
            )
        if query.category == "per_sequence":
            return tg._lookup_per_sequence(
                self.perf_db,
                query.layer_name,
                query.tp_size,
                shape["sequences"],
            )
        if query.category == "attention":
            return tg._lookup_attention_with_skew(
                self.perf_db,
                query.tp_size,
                shape["prefill_chunk"],
                shape["kv_prefill"],
                shape["n_decode"],
                shape["kv_decode_mean"],
                shape["kv_decode_max"],
                shape["kv_decode_min"],
            )
        if query.category == "moe":
            return tg._lookup_moe(
                self.perf_db,
                shape["tokens"],
                shape["activated_experts"],
            )
        raise KeyError(
            f"Unsupported HBF profile category {query.category!r}"
        )

    def describe(self):
        value = super().describe()
        value["provenance"] = dict(self.provenance)
        value["profile_root"] = self.perf_db["profile_root"]
        return value
