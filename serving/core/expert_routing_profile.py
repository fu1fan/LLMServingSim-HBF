import hashlib
import json
import math
from dataclasses import dataclass


_DISTRIBUTION_KIND = "marginal_histogram"
_LAYER_MAPPING = "seeded_permutation"
_SAMPLER = "plackett_luce_gumbel_topk"


@dataclass(frozen=True)
class ExpertRoutingProfile:
    profile_id: str
    target_model: str
    num_experts: int
    top_k: int
    source_counts: tuple
    selection_weights: tuple
    reference_tokens: int
    layer_mapping: str
    sampler: str
    calibration: dict
    sha256: str
    path: str


def validate_expert_routing_options(policy, profile_path, enable_block_copy):
    policy = str(policy).upper()
    if policy == "CUSTOM":
        if not profile_path:
            raise ValueError(
                "--expert-routing-profile is required when "
                "--expert-routing-policy=CUSTOM"
            )
        if enable_block_copy:
            raise ValueError(
                "CUSTOM expert routing requires --no-enable-block-copy "
                "because the calibrated profile varies by layer"
            )
    elif profile_path:
        raise ValueError(
            "--expert-routing-profile may only be used with "
            "--expert-routing-policy=CUSTOM"
        )


def _require_number(mapping, key, positive=False, nonnegative=False):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"'{key}' must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"'{key}' must be finite")
    if positive and value <= 0:
        raise ValueError(f"'{key}' must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"'{key}' must be non-negative")
    return value


def _load_vector(distribution, key, expected_len, positive):
    values = distribution.get(key)
    if not isinstance(values, list) or len(values) != expected_len:
        raise ValueError(
            f"'{key}' must contain exactly {expected_len} values"
        )
    result = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{key}[{idx}]' must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"'{key}[{idx}]' must be finite")
        if positive and value <= 0:
            raise ValueError(f"'{key}[{idx}]' must be positive")
        if not positive and value < 0:
            raise ValueError(f"'{key}[{idx}]' must be non-negative")
        result.append(value)
    if not positive and sum(result) <= 0:
        raise ValueError(f"'{key}' must contain at least one positive value")
    return tuple(result)


def load_expert_routing_profile(path, target_model, num_experts, top_k):
    with open(path, "rb") as f:
        payload = f.read()
    sha256 = hashlib.sha256(payload).hexdigest()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse expert routing profile '{path}': {exc}"
        ) from exc

    if data.get("schema_version") != 1:
        raise ValueError("Expert routing profile schema_version must be 1")

    profile_id = data.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("Expert routing profile requires a non-empty profile_id")

    profile_model = data.get("target_model")
    if profile_model != target_model:
        raise ValueError(
            f"Expert routing profile model mismatch: profile={profile_model!r}, "
            f"runtime={target_model!r}"
        )

    profile_experts = data.get("num_experts")
    profile_top_k = data.get("top_k")
    if profile_experts != int(num_experts):
        raise ValueError(
            f"Expert routing profile expert-count mismatch: "
            f"profile={profile_experts}, runtime={num_experts}"
        )
    if profile_top_k != int(top_k):
        raise ValueError(
            f"Expert routing profile top-k mismatch: "
            f"profile={profile_top_k}, runtime={top_k}"
        )

    distribution = data.get("distribution")
    if not isinstance(distribution, dict):
        raise TypeError("'distribution' must be an object")
    if distribution.get("kind") != _DISTRIBUTION_KIND:
        raise ValueError(
            f"'distribution.kind' must be {_DISTRIBUTION_KIND!r}"
        )
    if distribution.get("layer_mapping") != _LAYER_MAPPING:
        raise ValueError(
            f"'distribution.layer_mapping' must be {_LAYER_MAPPING!r}"
        )
    if distribution.get("sampler") != _SAMPLER:
        raise ValueError(
            f"'distribution.sampler' must be {_SAMPLER!r}"
        )

    source_counts = _load_vector(
        distribution, "source_counts", int(num_experts), positive=False
    )
    selection_weights = _load_vector(
        distribution, "selection_weights", int(num_experts), positive=True
    )
    reference_tokens = distribution.get("reference_tokens")
    if (
        isinstance(reference_tokens, bool)
        or not isinstance(reference_tokens, int)
        or reference_tokens <= 0
    ):
        raise ValueError("'reference_tokens' must be a positive integer")

    calibration = data.get("calibration")
    if not isinstance(calibration, dict):
        raise TypeError("'calibration' must be an object")
    for key in (
        "target_cv",
        "target_gini",
        "target_top5_share",
        "target_top10_share",
    ):
        _require_number(calibration, key, nonnegative=True)
    for key in ("source_url", "source_scope", "evidence_level"):
        if not isinstance(calibration.get(key), str) or not calibration[key].strip():
            raise ValueError(f"'calibration.{key}' must be a non-empty string")

    return ExpertRoutingProfile(
        profile_id=profile_id,
        target_model=profile_model,
        num_experts=int(profile_experts),
        top_k=int(profile_top_k),
        source_counts=source_counts,
        selection_weights=selection_weights,
        reference_tokens=reference_tokens,
        layer_mapping=distribution["layer_mapping"],
        sampler=distribution["sampler"],
        calibration=dict(calibration),
        sha256=sha256,
        path=str(path),
    )
