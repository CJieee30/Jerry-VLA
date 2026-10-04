"""Shared protocol for Drop-6 + task-conditional extra-3 on v044.

Universal skip U is Fisher Drop-6. Extra-3 is scored on the already-skipped
graph over the remaining 12 layers, including L0/L7. Whether L0/L7 are
universal-critical is read off the extra-3 histogram, not hard-coded.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

N_LAYERS = 18
EXTRA_K = 3
UNIVERSAL_DROP = [1, 13, 14, 15, 16, 17]
SUSPECTED_UNIVERSAL_KEEP = [0, 7]
REMAINING_12 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CANDIDATES = list(REMAINING_12)
GLOBAL_FISHER_EXTRA3 = [2, 3, 10]
GLOBAL_GREEDY_EXTRA3 = [2, 7, 11]
PER_TASK_NUM_BATCHES = 16
PER_TASK_BATCH_SIZE = 8
ATOMIC_SKILLS = ("pick", "place", "open", "close", "turn")
SKILL_NUM_BATCHES = 32
SKILL_BATCH_SIZE = 8
SKILL_BOUNDARY_MARGIN = 10
# Cross-suite cells that actually exist for pick/place/turn.
# Goal-pick excludes Goal5 (AtomicVLA labels push as pick).
SUITE_SKILL_CELLS = (
    ("Spatial", "pick"),
    ("Spatial", "place"),
    ("Object", "pick"),
    ("Object", "place"),
    ("Goal", "pick"),
    ("Goal", "place"),
    ("Goal", "turn"),
    ("Long", "pick"),
    ("Long", "place"),
    ("Long", "turn"),
)
SUITE_SKILL_EXCLUDE = (("Goal", "pick", "Goal5"),)
CORE2_HELDOUT_CELLS = (
    ("Goal", "turn"),
    ("Long", "turn"),
    ("Object", "place"),
    ("Spatial", "place"),
    ("Long", "place"),
)
CORE2_TURN_CELLS = (("Goal", "turn"), ("Long", "turn"))
CORE2_PLACE_CELLS = (("Object", "place"), ("Spatial", "place"), ("Long", "place"))
CORE2_TURN_MASKS = {
    "goal_like": [2, 7],
    "long_like": [2, 9],
}
CORE2_PLACE_MASKS = {
    "object_like": [2, 8],
    "spatial_long_like": [2, 7],
}
CORE2_ANCHORS = {
    "U_only": [],
    "L2_only": [2],
}
CORE2_GAP_MIN = 0.01
CORE2_HELDOUT_SEEDS = (44, 45)
CORE2_EXCLUDE_CALIB_SEEDS = (42, 43)
DEFAULT_ATOMIC_JSON = Path("/home/guorj/AtomicVLA/data_split_json/libero_lerobot.json")
DEFAULT_SKILL_INDEX_DIR = Path(__file__).resolve().parents[2] / "results" / "calib" / "skill_index"
TASK_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "dtr_drop9_recovered"
    / "results"
    / "task_layer"
    / "map_oracle_k1.json"
)
DEFAULT_CALIB_DIR = Path(__file__).resolve().parents[2] / "results" / "calib" / "per_task_16x8_seed42"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "drop6_extra3"
DEFAULT_MAP_DIR = DEFAULT_OUT_DIR / "maps"
EVAL_SHUFFLE_SEED = 1000
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABEL = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}


def dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def tasks_by_suite(catalog: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {s: [] for s in SUITE_ORDER}
    for row in catalog:
        suite = str(row["suite"])
        if suite not in grouped:
            raise RuntimeError(f"unknown suite {suite}")
        grouped[suite].append(row)
    for suite, rows in grouped.items():
        if len(rows) != 10:
            raise RuntimeError(f"{suite} has {len(rows)} tasks, expected 10")
    return grouped


def stamp_pooled_extra(
    catalog: list[dict[str, Any]],
    extra_for_task: dict[tuple[str, int], list[int]],
    extra_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in catalog:
        extra = extra_for_task[(str(row["suite"]), int(row["task_id"]))]
        rec = compact_task_row({**row, "extra_drop": extra})
        if extra_meta:
            rec.update(extra_meta)
        out.append(rec)
    return out


def load_task_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    src = Path(path) if path is not None else TASK_CATALOG_PATH
    payload = json.loads(src.read_text())
    tasks = []
    for row in payload["tasks"]:
        tasks.append(
            {
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "task": str(row["task"]),
                "short": str(row["short"]),
            }
        )
    if len(tasks) != 40:
        raise RuntimeError(f"{src} has {len(tasks)} tasks, expected 40")
    return tasks


def calib_path(calib_dir: Path, short: str) -> Path:
    return Path(calib_dir) / f"{short}.pt"


def skill_calib_dir(seed: int, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "results" / "calib"
    return base / f"per_skill_{SKILL_NUM_BATCHES}x{SKILL_BATCH_SIZE}_seed{seed}"


def suite_skill_stem(suite_label: str, verb: str) -> str:
    return f"{str(suite_label).lower()}_{str(verb)}"


def suite_skill_calib_dir(seed: int, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "results" / "calib"
    return base / f"suite_skill_{SKILL_NUM_BATCHES}x{SKILL_BATCH_SIZE}_seed{seed}"


def suite_skill_heldout_calib_dir(seed: int, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "results" / "calib"
    return base / f"suite_skill_heldout_{SKILL_NUM_BATCHES}x{SKILL_BATCH_SIZE}_seed{seed}"


def suite_skill_cell_name(suite_label: str, verb: str) -> str:
    return f"{suite_label}/{verb}"


def extra3_vs_global(extra: list[int], global_extra: list[int] | None = None) -> dict[str, Any]:
    extra_set = {int(x) for x in extra}
    glob = {int(x) for x in (global_extra if global_extra is not None else GLOBAL_GREEDY_EXTRA3)}
    inter = extra_set & glob
    union = extra_set | glob
    return {
        "extra_drop": sorted(extra_set),
        "global_greedy_extra3": sorted(glob),
        "intersection": sorted(inter),
        "symdiff": sorted(extra_set ^ glob),
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "differs": extra_set != glob,
    }


def wait_for_file(path: Path, *, timeout_s: float = 12 * 3600, poll_s: float = 10.0) -> Path:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file() and path.stat().st_size > 1024 and not tmp.exists():
            return path
        time.sleep(poll_s)
    raise TimeoutError(f"timed out waiting for {path}")


def extra3_from_fisher(scores: list[float]) -> dict[str, Any]:
    if len(scores) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} Fisher scores, got {len(scores)}")
    if any(i in UNIVERSAL_DROP for i in CANDIDATES):
        raise RuntimeError(f"candidates overlap U: {CANDIDATES}")
    cand_order = sorted(CANDIDATES, key=lambda i: (float(scores[i]), i))
    extra = sorted(cand_order[:EXTRA_K])
    if any(i in UNIVERSAL_DROP for i in extra):
        raise RuntimeError(f"Fisher extra-3 leaked universal drop layers: {extra}")
    ranks = {i: cand_order.index(i) + 1 for i in SUSPECTED_UNIVERSAL_KEEP}
    return {
        "extra_drop": extra,
        "candidate_redundancy_order": cand_order,
        "candidate_scores": {str(i): float(scores[i]) for i in CANDIDATES},
        "contains_l0": 0 in extra,
        "contains_l7": 7 in extra,
        "suspected_keep_ranks": ranks,
        "matches_global_fisher_extra3": extra == GLOBAL_FISHER_EXTRA3,
        "symdiff_vs_global": sorted(set(extra) ^ set(GLOBAL_FISHER_EXTRA3)),
    }


def protocol_dict() -> dict[str, Any]:
    return {
        "checkpoint": "lerobot/pi05_libero_finetuned_v044",
        "universal_drop": list(UNIVERSAL_DROP),
        "suspected_universal_keep": list(SUSPECTED_UNIVERSAL_KEEP),
        "candidates": list(CANDIDATES),
        "extra_k": EXTRA_K,
        "global_fisher_extra3": list(GLOBAL_FISHER_EXTRA3),
        "per_task_calib": f"{PER_TASK_NUM_BATCHES}x{PER_TASK_BATCH_SIZE}",
        "seed": 42,
        "skip": "skip_llm_layers / DTR Appendix B; score on skip(U) graph",
        "note": "U is locked from global Fisher Drop-6. PPL must not redefine U. extra-3 is chosen from the remaining 12 layers including L0/L7.",
    }


def summarize_extra_map(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    triples = [tuple(int(x) for x in row["extra_drop"]) for row in tasks]
    hist = Counter(triples)
    n = len(triples)
    n_match = sum(1 for t in triples if list(t) == GLOBAL_FISHER_EXTRA3)
    n_sym = sum(1 for t in triples if set(t) ^ set(GLOBAL_FISHER_EXTRA3))
    n_unique = len(hist)
    n_l0 = sum(1 for t in triples if 0 in t)
    n_l7 = sum(1 for t in triples if 7 in t)
    differentiate = (n_match <= 32 and n_unique >= 3) or n_sym >= 8
    return {
        "n_tasks": n,
        "n_unique_triples": n_unique,
        "n_match_global_fisher_extra3": n_match,
        "n_symdiff_ge1": n_sym,
        "n_extra3_contains_l0": n_l0,
        "n_extra3_contains_l7": n_l7,
        "histogram": {str(list(k)): int(v) for k, v in hist.most_common()},
        "should_run_closed_loop": differentiate,
        "gate": "main triple <=32/40 and >=2 other triples, or >=8/40 with symmetric difference >=1 vs [2,3,10]",
    }


def write_summary_md(method: str, payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        f"# v044 Drop-6 + extra-3 ({method})",
        "",
        "Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip "
        f"`U={UNIVERSAL_DROP}`. Candidates `{CANDIDATES}`. Per-task calib "
        f"{PER_TASK_NUM_BATCHES}×{PER_TASK_BATCH_SIZE}, seed 42.",
        "",
        f"- Unique extra-3 triples: **{summary['n_unique_triples']}**",
        f"- Match global Fisher `{{2,3,10}}`: **{summary['n_match_global_fisher_extra3']}/40**",
        f"- Symmetric difference ≥1 vs `{{2,3,10}}`: **{summary['n_symdiff_ge1']}/40**",
        f"- extra-3 contains L0: **{summary.get('n_extra3_contains_l0', 0)}/40**",
        f"- extra-3 contains L7: **{summary.get('n_extra3_contains_l7', 0)}/40**",
        f"- Closed-loop gate: **{summary['should_run_closed_loop']}**",
        "",
        "| Triple | Count |",
        "| --- | ---: |",
    ]
    for triple, count in summary["histogram"].items():
        lines.append(f"| `{triple}` | {count} |")
    lines += ["", "## Per-task extra-3", ""]
    for row in payload["tasks"]:
        lines.append(
            f"- `{row['short']}` ({row['suite']} {row['task_id']}): `{row['extra_drop']}`"
        )
    lines += ["", f"JSON: `{payload.get('out_json', path.with_suffix('.json'))}`", ""]
    path.write_text("\n".join(lines) + "\n")


def _normalize_prompt(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def compact_task_row(row: dict[str, Any]) -> dict[str, Any]:
    extra = [int(x) for x in row["extra_drop"]]
    bad = [i for i in extra if i not in CANDIDATES]
    if bad:
        raise ValueError(f"extra {extra} not in remaining-12 candidates {CANDIDATES}; leaked {bad}")
    if len(extra) != EXTRA_K:
        raise ValueError(f"expected extra-{EXTRA_K}, got {extra}")
    return {
        "suite": str(row["suite"]),
        "task_id": int(row["task_id"]),
        "task": str(row["task"]),
        "short": str(row["short"]),
        "extra_drop": extra,
        "union_drop": sorted(set(UNIVERSAL_DROP) | set(extra)),
    }


def make_eval_map(
    *,
    kind: str,
    method: str,
    tasks: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact = [compact_task_row(row) for row in tasks]
    if len(compact) != 40:
        raise RuntimeError(f"{kind}: expected 40 tasks, got {len(compact)}")
    payload = {
        "kind": kind,
        "method": method,
        "k": 9,
        "checkpoint": "lerobot/pi05_libero_finetuned_v044",
        "universal_drop": list(UNIVERSAL_DROP),
        "remaining_layers": list(CANDIDATES),
        "extra_k": EXTRA_K,
        "eval_seed": 1000,
        "compile_model": False,
        "n_action_steps": 10,
        "skip": "skip_llm_layers / DTR Appendix B prefix K/V; drop = U ∪ extra3(task)",
        "tasks": compact,
        "summary": summarize_extra_map(compact),
    }
    if extra:
        payload.update(extra)
    return payload


def shuffle_eval_map(src: dict[str, Any], *, seed: int = EVAL_SHUFFLE_SEED) -> dict[str, Any]:
    extras = [list(row["extra_drop"]) for row in src["tasks"]]
    shuffled = [list(x) for x in extras]
    random.Random(seed).shuffle(shuffled)
    n_same = sum(a == b for a, b in zip(extras, shuffled))
    tasks = []
    for row, extra in zip(src["tasks"], shuffled):
        rec = dict(row)
        rec["extra_drop"] = extra
        rec["union_drop"] = sorted(set(UNIVERSAL_DROP) | set(extra))
        tasks.append(rec)
    return make_eval_map(
        kind=f"{src['kind']}_shuffled",
        method=src["method"],
        tasks=tasks,
        extra={
            "shuffle_seed": seed,
            "n_same_as_source": n_same,
            "source_kind": src["kind"],
            "note": (
                f"Fisher-Yates permutation of the 40 {src['kind']} extra-3 triples "
                f"(same multiset, wrong tasks). seed={seed}."
            ),
        },
    )


def index_eval_map(payload: dict[str, Any]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], list[int]] = {}
    by_prompt: dict[str, list[int]] = {}
    for row in payload["tasks"]:
        extra = [int(x) for x in row["extra_drop"]]
        by_key[(str(row["suite"]), int(row["task_id"]))] = extra
        by_prompt[_normalize_prompt(row["task"])] = extra
    payload["_by_key"] = by_key
    payload["_by_prompt"] = by_prompt
    return payload


def load_eval_map(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if "tasks" not in payload:
        raise ValueError(f"{path} has no tasks[]")
    return index_eval_map(payload)


def lookup_extra(
    mapping: dict[str, Any],
    *,
    suite: str | None = None,
    task_id: int | None = None,
    prompt: str | None = None,
) -> list[int]:
    if suite is not None and task_id is not None:
        extra = mapping["_by_key"].get((str(suite), int(task_id)))
        if extra is not None:
            return list(extra)
    if prompt is not None:
        extra = mapping["_by_prompt"].get(_normalize_prompt(prompt))
        if extra is not None:
            return list(extra)
    raise KeyError(f"no extra_drop for suite={suite!r} task_id={task_id!r} prompt={prompt!r}")


def union_drop(extra: list[int], universal: list[int] | None = None) -> list[int]:
    base = list(universal if universal is not None else UNIVERSAL_DROP)
    return sorted(set(base) | {int(i) for i in extra})
