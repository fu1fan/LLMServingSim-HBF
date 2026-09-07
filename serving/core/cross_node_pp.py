"""Scope checks for the matrix builder's analytical cross-node PP model.

A spanning instance retains a logical host domain. Network timing is modeled,
but host-tier offload, shared prefix pools, PD and physical power accounting
must not silently reuse that logical host as a physical server.
"""


def validate_cross_node_pp_scope(cluster, *, prefix_storage="None",
                                 prefix_sharing=False, local_offloading=False,
                                 attn_offloading=False):
    layout = cluster.get("physical_layout", {})
    if not layout.get("cross_node_pp"):
        return
    nodes = cluster["nodes"]
    instances = [i for node in nodes for i in node["instances"]]
    unsupported = (
        prefix_storage not in (None, "None") or prefix_sharing
        or local_offloading or attn_offloading
        or any("power" in n for n in nodes)
        or any(i.get("pd_type") is not None
               or i.get("enable_local_offloading", False)
               or i.get("enable_attn_offloading", False) for i in instances)
    )
    if unsupported:
        raise ValueError(
            "Analytical cross-node PP uses logical host memory domains; "
            "host offload, secondary/shared prefix pools, PD and node power "
            "require a physical per-stage host-memory implementation"
        )
