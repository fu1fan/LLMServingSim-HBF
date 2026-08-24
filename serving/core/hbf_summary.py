"""Reproducible runtime metadata for HBF experiments."""
import json
import os


def build_hbf_runtime_summary(instances, schedulers, num_devices, run_id):
    memory_rows = []
    for instance_id, (instance, scheduler) in enumerate(zip(instances, schedulers)):
        memory = scheduler.memory
        hbf_mem = instance.get("hbf_mem")
        performance = (hbf_mem or {}).get("performance") or {}
        source = performance.get("source")
        evidence_level = None
        if source == "scale":
            evidence_level = "sensitivity-analysis"
        elif source == "profile":
            evidence_level = "external-simulator-backed"
        memory_rows.append({
            "instance_id": instance_id,
            "model": instance["model_name"],
            "hardware": instance["hardware"],
            "num_devices": instance["num_npus"],
            "tp_size": instance["tp_size"],
            "pp_size": instance["pp_size"],
            "ep_size": instance.get("ep_total", 1),
            "memory_scope": "per-rank",
            "hbm_capacity_bytes": memory.npu_mem,
            "hbm_weight_used_bytes": memory.hbm_weight,
            "hbm_kv_capacity_bytes": memory.npu_mem - memory.hbm_weight,
            "hbf_capacity_bytes": (memory.hbf_memory.capacity_bytes if memory.hbf_memory is not None else 0),
            "hbf_weight_used_bytes": memory.hbf_weight,
            "weight_residency_by_pp_rank": memory.weight_residency_by_pp_rank,
            "timing_source": source,
            "latency_scale": performance.get("latency_scale"),
            "profile_root": performance.get("profile_root"),
            "profile_hardware": performance.get("profile_hardware"),
            "evidence_level": evidence_level,
        })
    return {
        "schema_version": 1,
        "run_id": run_id,
        "num_devices": num_devices,
        "hbf_energy_unmodeled": any(instance.get("hbf_mem") is not None for instance in instances),
        "instances": memory_rows,
    }


def write_json_output(path, value):
    if not os.path.isabs(path):
        path = os.path.join("..", path)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
