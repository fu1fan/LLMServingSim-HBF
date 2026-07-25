# LLMCompass HBF Profile 运行指南

本文说明 LLMServingSim 如何消费 LLMCompass 导出的 HBF GPU
`runtime_ready` Profile v2，以及当前分层策略真正进入运行主链路的范围。
普通 GPU 的原有 Profile v1 和调度路径保持默认行为。

## 1. 分工与计费边界

HBF 模式没有把算子 demand 访存重新展开成 ASTRA 内存节点，而是按下面的
边界分工：

| 子系统 | 负责内容 |
| --- | --- |
| LLMCompass Profile v2 | 算子计算、HBM demand 读写、HBF demand 读写 |
| LLMServingSim | 请求调度、容量、对象驻留、Profile 场景选择、显式迁移决策 |
| ASTRA-Sim | TP/PP/EP 等原有通信，以及迁移、预取、淘汰等显式数据搬运 |

Profile 中每个 `time_us` 已经合成同一算子的计算时间和四向 demand
内存时间。CSV 同时保留以下审计列：

```text
hbm_read_bytes,hbm_write_bytes,hbf_read_bytes,hbf_write_bytes
```

这些字节用于检查和插值审计，不会再次生成迁移节点。只有对象驻留发生
变化时，Serving 才生成源端读和目标端写，例如：

```text
HBF -> HBM = HBF read + HBM write
HBM -> HBF = HBM read + HBF write
```

因此，`latency_accounting.includes` 必须包含 `compute`、
`hbm_demand_access`、`hbf_demand_access`，并排除 `migration`、
`prefetch`、`eviction`、`network_collective`。否则加载器会拒绝
Profile，以避免 demand 访存或 collective 被重复计费。

## 2. Profile v2 输入契约

默认情况下，实例按下面的路径查找 Profile：

```text
profiler/perf/<hardware>/<model_name>/<variant>/<memory_profile_id>/
├── meta.yaml
└── tp<N>/
    ├── dense.csv
    ├── per_sequence.csv
    ├── attention.csv
    └── moe.csv                 # 仅完整 MoE Profile
```

`variant` 由 `dtype` 和 `kv_cache_dtype` 推导。例如
`bfloat16 + auto` 对应 `bf16`。不要手工改名 LLMCompass 生成的
`memory_profile_id`，运行就绪 ID 与 performance identity 摘要绑定。

若 LLMCompass 将 bundle 导出到仓库外，可通过 `--profile-root` 直接指定
包含 `<hardware>/<model_name>/<variant>` 的 `perf` 根目录：

```bash
python -m serving \
  --profile-root /absolute/path/to/llmcompass-export/perf \
  --cluster-config /absolute/path/to/hbf-cluster.json \
  ...
```

相对 `--profile-root` 以 LLMServingSim 仓库根目录为基准。该参数同时用于
HBF manifest 校验和 Trace 性能查询，不需要再向仓库的 `profiler/perf`
创建软链接。

HBF 实例要求 manifest 至少满足：

- `profile_schema_version: 2`；
- `bundle_readiness: runtime_ready`；
- `runtime_compatible: true`；
- `scenario_binding: producer_verified_v1`；
- `calibration.schema: llmcompass_hbm_calibration_v1` 且
  `calibration.acceptance_passed: true`；
- `performance_basis.hbm: measured_calibrated`；
- `performance_basis.hbf: parameterized_projection`；
- `latency_accounting.demand_access_included: true`；
- 完整的 `architecture_requirements`、`engine_effective`、
  `attention_grid` 和目标 `tp<N>`；
- `memory_integration.mode` 为 `cli` 或 `csi`；
- `memory_integration.parameters` 使用 `schema_version: 1`、
  `timing_model: directional_v1`、
  `bandwidth_scope: pure_direction_effective`；
- HBM/HBF 均使用 `read_write_service: time_shared`，四个方向均使用
  `latency_scope: per_stream`；
- `fabric_model: none` 且独立 fabric 带宽为 `null`；
- 顶层 `access_catalog` 与 performance identity 中的副本完全一致；
- 每个 `scenario_catalog.<id>.accesses` 精确覆盖全部
  `access_catalog` 键。

`access_catalog` 的键固定为 `operator_id/access_id`，值必须且只能描述：

```yaml
attention/key_cache:
  semantic: kv_cache
  access_type: read
  lifetime: request
```

运行时不会由调用方指定一个场景名称。Scheduler 先冻结
`BatchMemoryView`，然后根据 `access_catalog` 把实际权重和 KV 驻留映射
到严格场景。当前只允许 `weight` 和 `kv_cache` 进入 HBF；
activation、temporary 和 output 等非持久对象保持在 HBM。若实际可达的
权重/KV 组合在 `scenario_catalog` 中没有精确场景，仿真会直接失败。

Profile v2 还采用严格运行边界：

- 运行时 `block_size` 必须与 `engine_effective.block_size` 相同；
- `max_num_batched_tokens` 和 `max_num_seqs` 不得超过验证范围；
- Attention 查询不得超过 `attention_grid.max_kv`；
- 运行所需 TP 必须有对应 `tp<N>` 目录；
- Dense、per-sequence、Attention 和可选 MoE 的活动 canonical 层必须
  与 `architecture_requirements` 完全一致。

## 3. 集群配置

可复制
[`configs/cluster/hbf_profile_example.json`](../configs/cluster/hbf_profile_example.json)
作为起点。该文件故意使用不存在的 hardware 和
`memory_profile_id` 占位符，仓库没有随附与它匹配的 HBF Profile。
其中所有容量、带宽和延迟数值也只是格式示例，不能作为硬件默认值。

运行前至少替换：

1. `hardware` 和 `performance_profile.memory_profile_id`；
2. `npu_mem.mem_size` 与 `hbf_mem.mem_size`；
3. `tp_size`、`num_npus`、`dtype`、`kv_cache_dtype`；
4. `block_size`、`max_num_batched_tokens`、`max_num_seqs`；
5. `link_bw`、`link_latency` 及请求数据集。

Profile 路径应能展开为：

```text
profiler/perf/
  <hardware>/<model_name>/<variant>/<memory_profile_id>/meta.yaml
```

HBF 配置的关键结构如下：

```json
{
  "hbf_mem": {
    "mem_size": 512
  },
  "performance_profile": {
    "mode": "memory_scenario_v2",
    "memory_profile_id": "replace-with-llmcompass-profile-id",
    "scenario_selection": "residency_derived"
  },
  "memory_tiering": {
    "weights": {
      "policy": "static_map",
      "default_tier": "hbm",
      "layers": {
        "lm_head": "hbf"
      },
      "blocks": [
        {
          "blocks": "0-15",
          "tier": "hbf"
        }
      ]
    },
    "kv": {
      "policy": "length_threshold",
      "admission_tier": "hbm",
      "threshold_tokens": 4096
    },
    "prefix": {
      "policy": "hbm_only"
    },
    "transfer": {
      "prefetch": "none",
      "capacity_fallback": "reject"
    },
    "communication_buffers": {
      "tier": "hbm",
      "allow_hbf_staging": false
    }
  }
}
```

`hbf_mem` 当前只接受以 GiB 为单位的 `mem_size`。不要在这里填写 HBF
带宽或延迟；HBM/HBF 各方向的带宽与固定延迟来自已验证 Profile 的
`memory_integration.parameters`，并用于生成本次运行的 ASTRA memory
配置。

`performance_profile.scenario_selection` 必须是
`residency_derived`，且不能再配置调用方声明的 `scenario_policy`。
旧 `placement` 仍保留给原有框架，但 Profile v2 要求 weights 和
`kv_loc` 均为 NPU/LOCAL，避免旧 offload 路径与 Profile 内 demand
访存重叠。

## 4. CLI 与 CSI 显式迁移资源

LLMServingSim 从 Profile 读取 HBM/HBF 的 read/write
`bandwidth_byte_per_second` 和 `fixed_latency_second`，转换为 ASTRA
使用的 GB/s 与 ns。集群 JSON 只提供容量，不重复声明这些性能参数。
Profile 内的 `memory_integration.parameters.integration_mode` 还必须与
顶层 `memory_integration.mode` 一致。ASTRA 当前使用整数 GB/s 和 ns，
适配器采用最近整数进行量化；零固定延迟保持为零。

运行就绪导出器和 Serving 加载器都会拒绝 `per_request_batch`、
`distinct_shared_resource`、`embedded_in_joint_calibration` 等尚不能
等价执行的参数，而不是静默套用近似公式。

### CLI

`memory_integration.mode: cli` 为 HBM 和 HBF 生成独立服务域：

```text
HBM -> service-group: hbm-data
HBF -> service-group: hbf-data
```

HBM 与 HBF 的显式搬运可以按依赖关系并行服务；每个 tier 仍分别受其
read/write 参数约束。

### CSI

`memory_integration.mode: csi` 把两者放入同一个
`gpu-memory-data` 服务域，使 HBM 与 HBF 的显式传输竞争同一 GPU 侧
数据通路。独立 fabric 流水级尚未进入 ASTRA 主链，因此不能在运行就绪
Profile 中配置 fabric 带宽上限。

同一次仿真中的所有 HBF instance 必须使用相同 performance identity
和内存集成参数。当前也禁止普通 GPU instance 与 HBF GPU instance
混用；普通或 HBF 可以分别做独立实验。

## 5. 分层策略的当前状态

配置解析采用严格字段检查。下面的“主链路可用”表示仅靠集群 JSON 和
CLI 即可进入当前 Scheduler/Trace/ASTRA 路径。

| 对象 | 策略 | 当前状态 |
| --- | --- | --- |
| 权重 | `hbm_only` | 主链路可用，全部静态驻留 HBM |
| 权重 | `hbf_only` | 主链路可用，全部静态驻留 HBF |
| 权重 | `static_map` | 主链路可用；优先级为 layer > block > default |
| 权重 | `hbf_backed_hbm_cache` | 策略引擎已有提升/LRU 接口；启动门禁拒绝，避免被误当成静态 HBF |
| KV | `hbm_only` | 主链路可用 |
| KV | `hbf_only` | 主链路可用 |
| KV | `length_threshold` | 主链路可用；以 request × transformer layer 为最小驻留单位，跨阈值时生成整层显式迁移 |
| KV | `watermark_lru` | 策略接口已有；CLI 主循环启动门禁拒绝 |
| Prefix | `hbm_only` | 与当前 RadixCache 主链路兼容 |
| Prefix | `hbf_only`、`hbf_backed_hbm_hot`、`instance_affinity` | 配置与策略接口已定义；启动门禁拒绝，尚未接入 RadixCache |
| Transfer | `prefetch: none` | 当前安全默认值 |
| Transfer | `next_layer`、`next_batch` | 已有配置契约；启动门禁拒绝，主循环尚未发起对应预取 |
| Transfer | `capacity_fallback` | 当前只允许 `reject`；CPU/CXL/HBF fallback 尚无完整四向迁移链 |
| 通信缓冲 | `tier: hbm` | 当前安全默认值，原有 collective 链路不变 |
| 通信缓冲 | `tier: hbf` | 配置接口已有；启动门禁拒绝，主循环尚未生成 staging 节点 |

若配置 `hbf_mem` 但省略 `memory_tiering`，所有持久对象仍默认在 HBM，
可用于验证普通 HBM 基线和 HBF Profile 加载契约。

## 6. 原有 Serving 特性的保留范围

| 特性 | 普通 GPU | HBF Profile v2 |
| --- | --- | --- |
| 动态到达、RR/RAND/LOAD 路由 | 保持原行为 | 保持；调度增加 HBM/HBF 容量检查 |
| 动态 batching、chunked prefill、混合 Prefill/Decode | 保持原行为 | 支持；必须落在 Profile engine 和 Attention 网格内 |
| TP | 保持原行为 | 支持 Profile 已覆盖的 TP；ALLREDUCE 仍由 ASTRA 计费 |
| PP | 保持原行为 | 原链路保留；权重容量按 PP stage 和 block 分账 |
| DP 同步 | 保持原行为 | 原同步链路保留；全部 HBF instance 必须使用同一性能身份 |
| P/D 分离 | 保持原行为 | 路由与原通信链路保留；当前强制 KV 为 `hbm_only`，HBF 权重仍可使用；tier-aware 跨实例 KV 交接尚未实现 |
| Dense Llama 3.x / Qwen3 | 保持原行为 | 当前 LLMCompass `runtime_ready` 导出与跨仓验收的主要运行范围 |
| MoE / EP / DP+EP | 保持原行为 | 只有完整 HBF MoE Profile 才可运行；必须声明 `moe_required` 并为全部场景提供 `moe.csv`。当前 Dense 导出不能替代它 |
| Prefix Caching / 多轮会话 | 保持原行为 | 会话依赖和路由保留；启用 RadixCache 时 KV 与 Prefix 策略必须均为 `hbm_only` |
| CPU/CXL Prefix 共享 | 保持原行为 | 仍属于旧 Radix/offload 路径，不等同于 HBF Prefix 策略 |
| local weight offload | 保持原行为 | Profile v2 启动门禁禁用 |
| PIM Attention offload | 保持原行为 | Profile v2 启动门禁禁用 |
| sub-batch interleaving | 保持原行为 | Profile v2 启动门禁禁用 |
| active-KV CPU/CXL 抢占 | 保持原行为 | 容量压力触发时 fail closed；旧 aggregate 行不能代替 HBM/HBF 四向显式迁移 |

Attention 遇到同一 batch 内 HBM/HBF 混合 KV 时，会按实际
request × layer 驻留分组，分别查询对应四维 Attention 性能，再合成为
当前层耗时；不需要扩展 Profile CSV 维度。

普通 GPU 配置不含 `hbf_mem` 时，分层功能关闭，权重和 KV 继续使用
原有 NPU/HBM 分账，Profile v1、调度、Prefix、offload 和并行框架不会
被 HBF 默认值改变。

HBF 模式启用 Prefix Caching 时，KV 与 Prefix 必须为 `hbm_only`；若
transformer 层数不能整除 PP，当前 RadixCache 的标量容量接口无法表达
非均匀 stage，启动门禁也会拒绝该组合。

## 7. 运行与检查

从 LLMServingSim 根目录执行：

```bash
python -m serving \
  --cluster-config configs/cluster/hbf_profile_example.json \
  --profile-root /absolute/path/to/llmcompass-export/perf \
  --dataset workloads/examples/replace-with-workload.jsonl \
  --memory-tiering-stats-output results/{run_id}-hbf.json \
  --network-backend analytical
```

`--memory-tiering-stats-output` 可选，支持 `{run_id}` 占位符。输出 JSON
按 instance 记录。相对路径以运行时的 `astra-sim` 工作目录为基准；
若需要固定输出位置，建议传入绝对路径。

- HBM/HBF 驻留与容量高水位；
- 显式迁移的方向、原因、对象类型和 transformer layer；
- 策略动作计数与 batch 驻留命中；
- Attention 查询中观察到的 HBM/HBF 驻留组。

每个实际形成的新 batch 只记录一次驻留命中或未命中；同一 batch 在其他
NPU 上重放不会重复计数，因此始终满足 `hits + misses == observed`。

该统计只接收 Serving 完成的显式迁移和驻留事件，不含 Profile 已计入
`time_us` 的 HBM/HBF demand 四向字节。需要审计算子 demand 流量时，应
查看 Profile CSV 的四个 audit 列，不能把两类字节相加后称为迁移量。

首次实验建议依次检查：

1. Profile 路径、identity、校准、TP 和场景覆盖通过加载门禁；
2. runtime 的 `block_size` 与 batch/sequence 上限不越界；
3. HBM 与 HBF 权重初始容量均可容纳实际静态映射；
4. workload 的最大 KV 长度不越过 Attention 网格；
5. Trace 中仅有显式迁移产生 HBM/HBF memory 节点；
6. 用相同 workload 分别运行全 HBM、静态 HBF 权重、HBF KV 阈值等
   对照组，不把 HBF 参数化预测描述成真实硬件测量。

路径和统计修复只提高运行可复现性与结果可审计性，不会把
`performance_basis.hbf: parameterized_projection` 提升为真实硬件测量。
