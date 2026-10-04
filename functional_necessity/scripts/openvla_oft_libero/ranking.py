"""High score = more important = kept. Redundancy order is the reverse."""

from __future__ import annotations

import torch

from paths import DROP_K


def importance_ranking(scores, k: int = DROP_K) -> dict:
    if torch.is_tensor(scores):
        values = [float(x) for x in scores.detach().cpu().flatten().tolist()]
    else:
        values = [float(x) for x in scores]
    n = len(values)
    keep_order = sorted(range(n), key=lambda i: values[i], reverse=True)
    redundancy_order = list(reversed(keep_order))
    keep_n = n - k
    drop = sorted(keep_order[keep_n:])
    keep = sorted(keep_order[:keep_n])
    return {
        "importance": values,
        "keep_order": keep_order,
        "redundancy_order": redundancy_order,
        "drop": drop,
        "keep": keep,
        "k": k,
        "n_layers": n,
    }


def print_ranking(method: str, scores) -> dict:
    report = importance_ranking(scores)
    print(f"\n{method} importance ({report['n_layers']} Llama layers; high = keep)")
    for i, value in enumerate(report["importance"]):
        print(f"  L{i:>2}: {value:.6e}")
    print("  redundancy-order: " + " > ".join(f"L{i}" for i in report["redundancy_order"]))
    print(f"  Drop-{report['k']}: {report['drop']}")
    return report
