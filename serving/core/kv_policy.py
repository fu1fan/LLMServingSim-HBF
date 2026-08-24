"""KV eviction policy: spill-to-HBF vs. drop-and-recompute cost model.

B5: when HBM KV capacity is exhausted, the scheduler may either
(a) *spill* a cold request's KV to a slower tier (HBF) and reload it later, or
(b) *recompute* the request's prefix on the next scheduling pass.

The correct choice is a cost comparison, not a fixed rule:

    spill_cost    = 2 * (spill_bytes / mem_bw + mem_latency)   # write + read back
    recompute_cost = recompute_tokens * cost_per_token          # prefill the prefix

``should_recompute`` returns True when recompute is cheaper. This mirrors the
real tradeoff the radix cache already makes implicitly (its ``_fit_for_insert``
drops locked-out prefixes for recompute), but makes it explicit and exposes it
as a tunable policy.
"""


def spill_cost_ns(spill_bytes, mem_bw_gb_s, mem_latency_ns):
    """Round-trip cost (write + read back) of spilling ``spill_bytes`` to HBF.

    ``mem_bw_gb_s`` is in GB/s; the division converts bytes → seconds → ns
    (1 GB/s == 1 byte/ns, so bytes / (GB/s) is already nanoseconds).
    """
    if spill_bytes <= 0:
        return 0.0
    if mem_bw_gb_s <= 0:
        raise ValueError("mem_bw_gb_s must be positive")
    if mem_latency_ns < 0:
        raise ValueError("mem_latency_ns must be non-negative")
    transfer_ns = spill_bytes / float(mem_bw_gb_s)
    return 2.0 * (transfer_ns + float(mem_latency_ns))


def recompute_cost_ns(recompute_tokens, cost_per_token_ns):
    """Prefill cost of recomputing ``recompute_tokens`` tokens."""
    if recompute_tokens < 0 or cost_per_token_ns < 0:
        raise ValueError("recompute_tokens and cost_per_token_ns must be non-negative")
    return recompute_tokens * float(cost_per_token_ns)


def should_recompute(
    spill_bytes,
    mem_bw_gb_s,
    mem_latency_ns,
    recompute_tokens,
    cost_per_token_ns,
):
    """Return True when dropping-and-recomputing beats spilling to HBF.

    ``cost_per_token_ns`` is the prefill cost of one token (TTFT-per-token
    derived from the operator profile / workload). Callers with no token-cost
    estimate should pass a large value to bias toward spilling (the safe
    default, since recompute is only cheaper for very hot, short prefixes).
    """
    if recompute_tokens <= 0:
        # Nothing to recompute → spilling would be pure waste.
        return True
    return recompute_cost_ns(recompute_tokens, cost_per_token_ns) < spill_cost_ns(
        spill_bytes, mem_bw_gb_s, mem_latency_ns
    )
