# Expert routing profiles

This directory stores versioned, data-only inputs for the simulator's
`CUSTOM` MoE expert-routing policy.

`qwen3_public_calibrated_v1.json` preserves the aggregate shape of a public
128-expert Qwen3-family routing example. The source reports 49,920 expert
assignments, corresponding to 6,240 top-8 tokens, with approximately:

- coefficient of variation: 0.55
- Gini coefficient: 0.30
- top-5 expert share: 9.9%
- top-10 expert share: 17.7%

The source does not fully disclose the exact checkpoint revision, layer, or
workload. The profile therefore calibrates a synthetic marginal distribution;
it is not a replay of Qwen3-235B-A22B production routing and does not preserve
the published expert IDs. The runtime deterministically permutes the load
shape per layer and samples token routes with weighted top-k without
replacement.

Use it with:

```bash
python -m serving \
  --expert-routing-policy CUSTOM \
  --expert-routing-profile configs/routing/qwen3_public_calibrated_v1.json \
  --expert-routing-seed 42 \
  --no-enable-block-copy \
  ...
```

The runtime fails if the target model, expert count, top-k, schema, or vector
length does not match the active model configuration.
