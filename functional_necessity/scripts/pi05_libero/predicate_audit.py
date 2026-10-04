"""Read-only BDDL goal-conjunct logging for closed-loop LIBERO eval.

Monkeypatches ``LiberoEnv.reset`` / ``step`` after the real physics step.
Does not change actions, rewards, or skip hooks. ``TurnOn`` is the same
predicate ``check_success`` already evaluates each step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("predicate_audit")

N_ACTION_STEPS = 10


def _bddl_env(libero_env):
    off = getattr(libero_env, "_env", None)
    if off is None:
        return None
    return getattr(off, "env", None)


def _flatten_goal(goal_state: Any) -> list[list[str]]:
    out: list[list[str]] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple)) and node:
            head = node[0]
            if isinstance(head, str) and head.lower() == "and":
                for child in node[1:]:
                    walk(child)
                return
            if isinstance(head, (list, tuple)):
                for child in node:
                    walk(child)
                return
            out.append([str(x) for x in node])
            return
        out.append([str(node)])

    walk(goal_state)
    return out


def _role(pred: list[str]) -> str:
    name = pred[0].lower() if pred else ""
    if name == "turnon":
        return "turn"
    if name == "close":
        return "close"
    if name in ("in", "on"):
        return "place"
    return name


def _conjunct_key(pred: list[str]) -> str:
    return ":".join(pred)


def _eval_conjuncts(bddl) -> dict[str, bool]:
    goal = _flatten_goal(bddl.parsed_problem.get("goal_state"))
    bits: dict[str, bool] = {}
    for pred in goal:
        key = _conjunct_key(pred)
        try:
            bits[key] = bool(bddl._eval_predicate(pred))
        except Exception as exc:  # noqa: BLE001 — keep eval running
            LOGGER.warning("predicate eval failed %s: %s", pred, exc)
            bits[key] = False
    return bits


def _classify(ep: dict[str, Any]) -> dict[str, Any]:
    by_role = {c["role"]: c for c in ep["conjuncts"]}
    place = by_role.get("place")
    close = by_role.get("close")
    turn = by_role.get("turn")
    out: dict[str, Any] = {}
    if place is not None:
        out["place_ever"] = place["ever"]
        out["place_end"] = place["end"]
        out["place_first_step"] = place["first_step"]
        out["place_first_chunk"] = place["first_chunk"]
    if close is not None:
        out["close_ever"] = close["ever"]
        out["close_end"] = close["end"]
        out["close_first_step"] = close["first_step"]
        out["close_first_chunk"] = close["first_chunk"]
    if turn is not None:
        out["turn_ever"] = turn["ever"]
        out["turn_end"] = turn["end"]
        out["turn_first_step"] = turn["first_step"]
        out["turn_first_chunk"] = turn["first_chunk"]
    if place is not None and close is not None:
        pe, ce = place["end"], close["end"]
        if pe and ce:
            out["failure_phase"] = "success"
        elif pe and not ce:
            out["failure_phase"] = "close"
        else:
            out["failure_phase"] = "place"
    elif turn is not None and place is not None:
        te, pe = turn["end"], place["end"]
        if te and pe:
            out["failure_phase"] = "success"
        elif not te and pe:
            out["failure_phase"] = "turn"
        elif te and not pe:
            out["failure_phase"] = "place"
        else:
            out["failure_phase"] = "both"
    return out


class PredicateAuditor:
    def __init__(self, *, n_action_steps: int = N_ACTION_STEPS) -> None:
        self.n_action_steps = int(n_action_steps)
        self.episodes: list[dict[str, Any]] = []
        self.cur: dict[str, Any] | None = None
        self._ep_counter = 0

    def _start(self, libero_env) -> None:
        bddl = _bddl_env(libero_env)
        goal = _flatten_goal(bddl.parsed_problem.get("goal_state")) if bddl is not None else []
        conjuncts = []
        for pred in goal:
            conjuncts.append(
                {
                    "pred": pred,
                    "key": _conjunct_key(pred),
                    "role": _role(pred),
                    "ever": False,
                    "end": False,
                    "first_step": None,
                    "first_chunk": None,
                    "last_true_step": None,
                }
            )
        self.cur = {
            "episode_index": self._ep_counter,
            "task": getattr(libero_env, "task", None),
            "task_id": getattr(libero_env, "task_id", None),
            "task_description": getattr(libero_env, "task_description", None),
            "env_step": 0,
            "success": False,
            "terminated": False,
            "conjuncts": conjuncts,
        }
        self._ep_counter += 1

    def _apply_bits(self, bits: dict[str, bool], step: int) -> None:
        assert self.cur is not None
        chunk = (step - 1) // self.n_action_steps
        for conj in self.cur["conjuncts"]:
            val = bool(bits.get(conj["key"], False))
            conj["end"] = val
            if val:
                conj["ever"] = True
                conj["last_true_step"] = step
                if conj["first_step"] is None:
                    conj["first_step"] = step
                    conj["first_chunk"] = chunk

    def _finalize(self, *, success: bool, terminated: bool) -> None:
        if self.cur is None:
            return
        self.cur["success"] = bool(success)
        self.cur["terminated"] = bool(terminated)
        self.cur["n_steps"] = int(self.cur["env_step"])
        self.cur["classification"] = _classify(self.cur)
        self.episodes.append(self.cur)
        self.cur = None

    def on_reset(self, libero_env) -> None:
        if self.cur is not None:
            self._finalize(success=bool(self.cur.get("success")), terminated=True)
        self._start(libero_env)

    def on_step(self, libero_env, info: dict[str, Any], done: bool) -> None:
        if self.cur is None:
            self._start(libero_env)
        self.cur["env_step"] += 1
        bddl = _bddl_env(libero_env)
        bits = _eval_conjuncts(bddl) if bddl is not None else {}
        self._apply_bits(bits, self.cur["env_step"])
        success = bool(info.get("is_success", False))
        if success:
            self.cur["success"] = True
        if done:
            self._finalize(success=success or bool(self.cur.get("success")), terminated=True)

    def dump(self, path: Path) -> dict[str, Any]:
        if self.cur is not None:
            if int(self.cur.get("env_step") or 0) == 0:
                self.cur = None
            else:
                self._finalize(success=bool(self.cur.get("success")), terminated=True)
        self.episodes = [ep for ep in self.episodes if int(ep.get("n_steps") or 0) > 0]
        phases: dict[str, int] = {}
        for ep in self.episodes:
            key = str(ep.get("classification", {}).get("failure_phase", "unknown"))
            phases[key] = phases.get(key, 0) + 1
        payload = {
            "kind": "bddl_predicate_audit",
            "n_action_steps": self.n_action_steps,
            "read_only": True,
            "n_episodes": len(self.episodes),
            "n_success": sum(1 for ep in self.episodes if ep.get("success")),
            "phase_counts": phases,
            "episodes": self.episodes,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        LOGGER.info("wrote predicate audit %s phase_counts=%s", path, phases)
        return payload


def install_predicate_audit(*, n_action_steps: int = N_ACTION_STEPS) -> PredicateAuditor:
    from lerobot.envs.libero import LiberoEnv

    auditor = PredicateAuditor(n_action_steps=n_action_steps)
    orig_reset = LiberoEnv.reset
    orig_step = LiberoEnv.step

    def reset(self, seed=None, **kwargs):
        obs, info = orig_reset(self, seed=seed, **kwargs)
        auditor.on_reset(self)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = orig_step(self, action)
        auditor.on_step(self, info, bool(terminated or truncated))
        return obs, reward, terminated, truncated, info

    LiberoEnv.reset = reset  # type: ignore[method-assign]
    LiberoEnv.step = step  # type: ignore[method-assign]
    LOGGER.info("installed read-only BDDL predicate audit on LiberoEnv")
    return auditor
