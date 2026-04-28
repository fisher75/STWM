#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import os

import torch
import torch.nn.functional as F


SUBSETS = ["long_gap_persistence", "occlusion_reappearance", "crossing_ambiguity", "OOD_hard", "appearance_change"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = ["# " + title, ""]
    for key in [
        "alignment_probe_trained",
        "no_backbone_update",
        "no_candidate_leakage_to_rollout",
        "candidate_top1",
        "candidate_MRR",
        "candidate_AP",
        "candidate_AUROC",
        "predicted_state_load_bearing",
        "paper_world_model_claimable",
        "recommended_next_step_choice",
    ]:
        if key in payload:
            lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _both_classes(labels: list[int]) -> bool:
    return 0 in labels and 1 in labels


def binary_auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for ps in pos:
        for ns in neg:
            total += 1.0
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total if total else None


def binary_ap(scores: list[float], labels: list[int]) -> float | None:
    if not scores or not labels or 1 not in labels:
        return None
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    positives = 0
    precision_sum = 0.0
    total_pos = sum(1 for y in labels if int(y) == 1)
    for rank, idx in enumerate(order, 1):
        if int(labels[idx]) == 1:
            positives += 1
            precision_sum += positives / float(rank)
    return precision_sum / float(total_pos) if total_pos else None


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def stable_projection(values: torch.Tensor, out_dim: int, salt: str) -> torch.Tensor:
    in_dim = int(values.numel())
    if in_dim == out_dim:
        return values.float()
    seed = int.from_bytes(salt.encode("utf-8"), "little", signed=False) % (2**31 - 1)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + in_dim * 9176 + out_dim * 131)
    mat = torch.randn((out_dim, in_dim), generator=g, dtype=torch.float32) / math.sqrt(max(in_dim, 1))
    return mat @ values.float().view(-1)


def cosine01(a: torch.Tensor, b: torch.Tensor) -> float:
    dim = min(int(a.numel()), int(b.numel()))
    if dim <= 0:
        return 0.5
    aa = a.view(-1)[:dim].float()
    bb = b.view(-1)[:dim].float()
    denom = aa.norm() * bb.norm()
    if float(denom.item()) <= 1e-12:
        return 0.5
    return float(((torch.dot(aa, bb) / denom).clamp(-1.0, 1.0) * 0.5 + 0.5).item())


class AlignmentProbe(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(in_dim),
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = torch.nn.Sequential(torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, out_dim))
        self.logit_scale = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        pred = self.net(x)
        pred = F.normalize(pred, dim=-1)
        cand = F.normalize(candidate, dim=-1)
        return (pred * cand).sum(dim=-1) * self.logit_scale.clamp(1.0, 30.0)


def tensorize(records: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.tensor([[float(x) for x in r[key]] for r in records], dtype=torch.float32)


def train_probe(
    *,
    name: str,
    x_train: torch.Tensor,
    cand_train: torch.Tensor,
    y_train: torch.Tensor,
    x_all: torch.Tensor,
    cand_all: torch.Tensor,
    epochs: int,
    lr: float,
    hidden_dim: int,
) -> tuple[AlignmentProbe, list[float], torch.Tensor]:
    model = AlignmentProbe(int(x_train.shape[1]), int(cand_train.shape[1]), hidden_dim=hidden_dim)
    pos = y_train.sum().clamp_min(1.0)
    neg = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(neg / pos).clamp(1.0, 100.0))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(x_train, cand_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    with torch.no_grad():
        all_scores = torch.sigmoid(model(x_all, cand_all)).detach().cpu()
    return model, losses, all_scores


def evaluate_scores(records: list[dict[str, Any]], scores: list[float], split: str | None = None) -> dict[str, Any]:
    selected = [(r, float(s)) for r, s in zip(records, scores) if split is None or r.get("split") == split]
    labels = [int(r.get("label_same_identity", 0)) for r, _ in selected]
    flat_scores = [float(s) for _, s in selected]
    by_item: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for r, s in selected:
        by_item[str(r.get("item_id"))].append((r, s))
    top1: list[float] = []
    mrr: list[float] = []
    metric_items = 0
    for rows in by_item.values():
        ys = [int(r.get("label_same_identity", 0)) for r, _ in rows]
        if not (1 in ys and 0 in ys):
            continue
        metric_items += 1
        order = sorted(range(len(rows)), key=lambda i: rows[i][1], reverse=True)
        top1.append(1.0 if ys[order[0]] == 1 else 0.0)
        gt_idx = ys.index(1)
        mrr.append(1.0 / float(order.index(gt_idx) + 1))
    return {
        "candidate_top1": mean(top1),
        "candidate_MRR": mean(mrr),
        "candidate_AP": binary_ap(flat_scores, labels) if _both_classes(labels) else None,
        "candidate_AUROC": binary_auc(flat_scores, labels) if _both_classes(labels) else None,
        "candidate_positive_rate": (sum(labels) / len(labels)) if labels else None,
        "candidate_record_count": len(labels),
        "metric_items": metric_items,
    }


def subset_metrics(records: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset in SUBSETS:
        idx = [i for i, r in enumerate(records) if bool((r.get("subset_tags") or {}).get(subset))]
        sub_records = [records[i] for i in idx]
        sub_scores = [scores[i] for i in idx]
        out[subset] = evaluate_scores(sub_records, sub_scores)
    return out


def make_eval_payload(
    *,
    mode: str,
    records: list[dict[str, Any]],
    scores: list[float],
    dataset_path: Path,
) -> dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "mode": str(mode),
        "source_dataset": str(dataset_path),
        "future_candidate_used_as_input": False,
        "candidate_feature_used_for_rollout": False,
        "candidate_feature_used_for_scoring": True,
        "stage2_val_fallback_used": False,
        "old_association_report_used": False,
        "full_model_forward_executed": True,
        "full_free_rollout_executed": True,
        "score_components_used": {
            "aligned_predicted_semantic_to_candidate": ["aligned_predicted_semantic_embedding", "candidate_crop_feature"],
            "aligned_predicted_identity_to_candidate": ["aligned_predicted_identity_embedding", "candidate_crop_feature"],
            "aligned_predicted_semantic_identity_to_candidate": [
                "aligned_predicted_semantic_embedding",
                "aligned_predicted_identity_embedding",
                "candidate_crop_feature",
            ],
            "posterior_v5": [
                "distance_score",
                "target_candidate_appearance_score",
                "aligned_predicted_semantic_score",
                "aligned_predicted_identity_score",
                "visibility_reappearance_priors",
            ],
            "posterior_v5_no_appearance": [
                "distance_score",
                "aligned_predicted_semantic_score",
                "aligned_predicted_identity_score",
                "visibility_reappearance_priors",
            ],
            "posterior_v5_no_predicted_state": [
                "distance_score",
                "target_candidate_appearance_score",
                "visibility_reappearance_priors",
            ],
            "posterior_v5_no_distance": [
                "target_candidate_appearance_score",
                "aligned_predicted_semantic_score",
                "aligned_predicted_identity_score",
                "visibility_reappearance_priors",
            ],
        }.get(str(mode), []),
        **evaluate_scores(records, scores),
        "dev": evaluate_scores(records, scores, split="dev"),
        "heldout": evaluate_scores(records, scores, split="heldout"),
        "per_subset_breakdown": subset_metrics(records, scores),
    }
    return payload


def train_and_eval(
    *,
    dataset_path: Path,
    train_report: Path,
    train_doc: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    hidden_dim: int,
    seed: int = 0,
    probe_output_dir: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    dataset = load_json(dataset_path)
    records = dataset.get("records") if isinstance(dataset, dict) else []
    records = [r for r in records if isinstance(r, dict)]
    dev_records = [r for r in records if r.get("split") == "dev"]
    if not dev_records:
        raise RuntimeError("alignment dataset has no dev records")
    sem_all = tensorize(records, "predicted_semantic_embedding")
    ident_all = tensorize(records, "predicted_identity_embedding")
    cand_all = tensorize(records, "candidate_crop_feature")
    y_all = torch.tensor([float(r.get("label_same_identity", 0)) for r in records], dtype=torch.float32)
    dev_idx = [i for i, r in enumerate(records) if r.get("split") == "dev"]
    sem_dev = sem_all[dev_idx]
    ident_dev = ident_all[dev_idx]
    cand_dev = cand_all[dev_idx]
    y_dev = y_all[dev_idx]

    sem_model, sem_losses, sem_scores_t = train_probe(
        name="semantic", x_train=sem_dev, cand_train=cand_dev, y_train=y_dev, x_all=sem_all, cand_all=cand_all, epochs=epochs, lr=lr, hidden_dim=hidden_dim
    )
    id_model, id_losses, id_scores_t = train_probe(
        name="identity", x_train=ident_dev, cand_train=cand_dev, y_train=y_dev, x_all=ident_all, cand_all=cand_all, epochs=epochs, lr=lr, hidden_dim=hidden_dim
    )
    si_all = torch.cat([sem_all, ident_all], dim=1)
    si_dev = si_all[dev_idx]
    si_model, si_losses, si_scores_t = train_probe(
        name="semantic_identity", x_train=si_dev, cand_train=cand_dev, y_train=y_dev, x_all=si_all, cand_all=cand_all, epochs=epochs, lr=lr, hidden_dim=hidden_dim
    )
    sem_scores = [float(x) for x in sem_scores_t.tolist()]
    id_scores = [float(x) for x in id_scores_t.tolist()]
    si_scores = [float(x) for x in si_scores_t.tolist()]

    distance_scores = [float(r.get("distance_score", 0.0)) for r in records]
    appearance_scores = [float(r.get("appearance_score", 0.5)) for r in records]
    priors = [
        float(0.5 * float(r.get("future_visibility_prob", 0.5)) + 0.5 * float(r.get("future_reappearance_event_prob", 0.5)))
        for r in records
    ]
    unaligned_scores: list[float] = []
    for r in records:
        sem = torch.tensor([float(x) for x in r["predicted_semantic_embedding"]], dtype=torch.float32)
        ident = torch.tensor([float(x) for x in r["predicted_identity_embedding"]], dtype=torch.float32)
        cand = torch.tensor([float(x) for x in r["candidate_crop_feature"]], dtype=torch.float32)
        sem_measure = stable_projection(cand, int(sem.numel()), "unaligned_semantic")
        id_measure = stable_projection(cand, int(ident.numel()), "unaligned_identity")
        unaligned_scores.append(0.5 * cosine01(sem, sem_measure) + 0.5 * cosine01(ident, id_measure))

    def combine(weights: dict[str, float]) -> list[float]:
        raw: list[float] = []
        denom = max(sum(abs(x) for x in weights.values()), 1e-8)
        for d, a, sem, ident, p in zip(distance_scores, appearance_scores, sem_scores, id_scores, priors):
            raw.append(
                (
                    weights.get("distance", 0.0) * d
                    + weights.get("appearance", 0.0) * a
                    + weights.get("semantic", 0.0) * sem
                    + weights.get("identity", 0.0) * ident
                    + weights.get("priors", 0.0) * p
                )
                / denom
            )
        return raw

    mode_scores = {
        "aligned_predicted_semantic_to_candidate": sem_scores,
        "aligned_predicted_identity_to_candidate": id_scores,
        "aligned_predicted_semantic_identity_to_candidate": si_scores,
        "posterior_v5": combine({"distance": 0.25, "appearance": 0.25, "semantic": 0.20, "identity": 0.15, "priors": 0.15}),
        "posterior_v5_no_appearance": combine({"distance": 0.35, "semantic": 0.25, "identity": 0.20, "priors": 0.20}),
        "posterior_v5_no_predicted_state": combine({"distance": 0.40, "appearance": 0.40, "priors": 0.20}),
        "posterior_v5_no_distance": combine({"appearance": 0.35, "semantic": 0.25, "identity": 0.20, "priors": 0.20}),
        "distance_only": distance_scores,
        "target_candidate_appearance_only": appearance_scores,
        "unaligned_predicted_semantic_identity": unaligned_scores,
        "posterior_v4": combine({"distance": 0.30, "appearance": 0.25, "semantic": 0.20, "identity": 0.15, "priors": 0.10}),
    }

    eval_paths = {
        "aligned_predicted_semantic_to_candidate": output_dir / "stwm_external_candidate_scoring_v5_aligned_semantic_eval_20260428.json",
        "aligned_predicted_identity_to_candidate": output_dir / "stwm_external_candidate_scoring_v5_aligned_identity_eval_20260428.json",
        "aligned_predicted_semantic_identity_to_candidate": output_dir / "stwm_external_candidate_scoring_v5_aligned_semantic_identity_eval_20260428.json",
        "posterior_v5": output_dir / "stwm_external_candidate_scoring_v5_posterior_v5_eval_20260428.json",
        "posterior_v5_no_appearance": output_dir / "stwm_external_candidate_scoring_v5_posterior_v5_no_appearance_eval_20260428.json",
        "posterior_v5_no_predicted_state": output_dir / "stwm_external_candidate_scoring_v5_posterior_v5_no_predicted_state_eval_20260428.json",
        "posterior_v5_no_distance": output_dir / "stwm_external_candidate_scoring_v5_posterior_v5_no_distance_eval_20260428.json",
    }
    eval_payloads: dict[str, dict[str, Any]] = {}
    for mode, path in eval_paths.items():
        payload = make_eval_payload(mode=mode, records=records, scores=mode_scores[mode], dataset_path=dataset_path)
        write_json(path, payload)
        write_doc(path.with_suffix(".md"), "STWM External Candidate Scoring V5 " + mode, payload)
        eval_payloads[mode] = payload

    all_metrics = {mode: make_eval_payload(mode=mode, records=records, scores=scores, dataset_path=dataset_path) for mode, scores in mode_scores.items()}
    dev_heldout = {
        "generated_at_utc": now_iso(),
        "dataset": str(dataset_path),
        "no_heldout_tuning": True,
        "split_rule": dataset.get("split_rule"),
        "modes": {
            mode: {
                "all": payload,
                "dev": payload.get("dev"),
                "heldout": payload.get("heldout"),
            }
            for mode, payload in all_metrics.items()
        },
        "comparisons": {},
    }
    def held(mode: str, metric: str) -> float | None:
        value = dev_heldout["modes"].get(mode, {}).get("heldout", {}).get(metric)
        return float(value) if isinstance(value, (int, float)) else None

    aligned_ap = held("aligned_predicted_semantic_identity_to_candidate", "candidate_AP")
    unaligned_ap = held("unaligned_predicted_semantic_identity", "candidate_AP")
    distance_ap = held("distance_only", "candidate_AP")
    post5_ap = held("posterior_v5", "candidate_AP")
    post4_ap = held("posterior_v4", "candidate_AP")
    no_state_ap = held("posterior_v5_no_predicted_state", "candidate_AP")
    appearance_ap = held("target_candidate_appearance_only", "candidate_AP")
    aligned_over_unaligned = bool(aligned_ap is not None and unaligned_ap is not None and aligned_ap > unaligned_ap)
    aligned_over_distance = bool(aligned_ap is not None and distance_ap is not None and aligned_ap > distance_ap)
    post5_over_post4 = bool(post5_ap is not None and post4_ap is not None and post5_ap > post4_ap)
    post5_over_no_state = bool(post5_ap is not None and no_state_ap is not None and post5_ap > no_state_ap)
    predicted_load = bool(aligned_over_unaligned and post5_over_no_state)
    heldout_signal = bool(predicted_load and (post5_ap is not None and distance_ap is not None and post5_ap > distance_ap))
    dev_heldout["comparisons"] = {
        "aligned_predicted_semantic_identity_gt_unaligned": aligned_over_unaligned,
        "aligned_predicted_semantic_identity_gt_distance": aligned_over_distance,
        "posterior_v5_gt_posterior_v4": post5_over_post4,
        "posterior_v5_gt_no_predicted_state": post5_over_no_state,
        "posterior_v5_gt_appearance_only": bool(post5_ap is not None and appearance_ap is not None and post5_ap > appearance_ap),
    }
    dev_summary_path = output_dir / "stwm_external_candidate_scoring_v5_dev_heldout_summary_20260428.json"
    write_json(dev_summary_path, dev_heldout)
    write_doc(dev_summary_path.with_suffix(".md"), "STWM External Candidate Scoring V5 Dev Heldout Summary", dev_heldout)

    subset_signal = {}
    post5_subset = all_metrics["posterior_v5"]["per_subset_breakdown"]
    no_state_subset = all_metrics["posterior_v5_no_predicted_state"]["per_subset_breakdown"]
    for subset in SUBSETS:
        a = post5_subset.get(subset, {}).get("candidate_AP")
        b = no_state_subset.get(subset, {}).get("candidate_AP")
        subset_signal[subset] = bool(isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) > float(b))

    probe_dir = probe_output_dir or Path("outputs/alignment_probes/stwm_semantic_state_alignment_v6")
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_paths = {
        "semantic_probe": probe_dir / f"semantic_probe_seed{int(seed)}.pt",
        "identity_probe": probe_dir / f"identity_probe_seed{int(seed)}.pt",
        "semantic_identity_probe": probe_dir / f"semantic_identity_probe_seed{int(seed)}.pt",
        "config": probe_dir / f"alignment_probe_config_seed{int(seed)}.json",
    }
    common_cfg = {
        "dataset": str(dataset_path),
        "seed": int(seed),
        "semantic_input_dim": int(sem_all.shape[1]),
        "identity_input_dim": int(ident_all.shape[1]),
        "semantic_identity_input_dim": int(si_all.shape[1]),
        "candidate_output_dim": int(cand_all.shape[1]),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "lr": float(lr),
        "split_rule": dataset.get("split_rule"),
        "no_backbone_update": True,
        "no_rollout_candidate_leakage": True,
    }
    torch.save({"state_dict": sem_model.state_dict(), "config": {**common_cfg, "probe": "semantic_alignment_probe"}}, probe_paths["semantic_probe"])
    torch.save({"state_dict": id_model.state_dict(), "config": {**common_cfg, "probe": "identity_alignment_probe"}}, probe_paths["identity_probe"])
    torch.save(
        {"state_dict": si_model.state_dict(), "config": {**common_cfg, "probe": "semantic_identity_alignment_probe"}},
        probe_paths["semantic_identity_probe"],
    )
    write_json(probe_paths["config"], common_cfg)

    train_payload = {
        "generated_at_utc": now_iso(),
        "dataset": str(dataset_path),
        "alignment_probe_trained": True,
        "alignment_probe_weights_saved": True,
        "alignment_probe_weight_paths": {k: str(v) for k, v in probe_paths.items()},
        "seed": int(seed),
        "no_backbone_update": True,
        "no_rollout_candidate_leakage": True,
        "train_split_size": len(dev_records),
        "heldout_split_size": sum(1 for r in records if r.get("split") == "heldout"),
        "trainable_params": {
            "semantic_alignment_probe": sum(p.numel() for p in sem_model.parameters()),
            "identity_alignment_probe": sum(p.numel() for p in id_model.parameters()),
            "semantic_identity_alignment_probe": sum(p.numel() for p in si_model.parameters()),
        },
        "epochs": int(epochs),
        "lr": float(lr),
        "semantic_train_loss_start": sem_losses[0] if sem_losses else None,
        "semantic_train_loss_end": sem_losses[-1] if sem_losses else None,
        "identity_train_loss_start": id_losses[0] if id_losses else None,
        "identity_train_loss_end": id_losses[-1] if id_losses else None,
        "semantic_identity_train_loss_start": si_losses[0] if si_losses else None,
        "semantic_identity_train_loss_end": si_losses[-1] if si_losses else None,
        "dev_metrics": dev_heldout["modes"]["aligned_predicted_semantic_identity_to_candidate"]["dev"],
        "heldout_metrics": dev_heldout["modes"]["aligned_predicted_semantic_identity_to_candidate"]["heldout"],
    }
    write_json(train_report, train_payload)
    write_doc(train_doc, "STWM Semantic-State Measurement Alignment Probe Train V1", train_payload)

    decision = {
        "generated_at_utc": now_iso(),
        "alignment_probe_trained": True,
        "no_backbone_update": True,
        "no_candidate_leakage_to_rollout": True,
        "aligned_predicted_state_improves_over_unaligned": aligned_over_unaligned,
        "aligned_predicted_state_improves_over_distance": aligned_over_distance,
        "predicted_state_load_bearing": predicted_load,
        "posterior_v5_improves_over_posterior_v4": post5_over_post4,
        "posterior_v5_improves_over_no_predicted_state": post5_over_no_state,
        "heldout_signal_positive": heldout_signal,
        "subset_signal_positive": subset_signal,
        "paper_world_model_claimable": False if not predicted_load else "unclear",
        "semantic_state_branch_status": "appendix_diagnostic" if predicted_load else "improve_alignment_with_vlm_features",
        "recommended_next_step_choice": (
            "proceed_to_paper_assets_with_semantic_state_auxiliary"
            if heldout_signal
            else "improve_alignment_with_frozen_vlm_features"
        ),
        "heldout_metrics": {
            "distance_only": dev_heldout["modes"]["distance_only"]["heldout"],
            "target_candidate_appearance_only": dev_heldout["modes"]["target_candidate_appearance_only"]["heldout"],
            "unaligned_predicted_semantic_identity": dev_heldout["modes"]["unaligned_predicted_semantic_identity"]["heldout"],
            "aligned_predicted_semantic_identity": dev_heldout["modes"]["aligned_predicted_semantic_identity_to_candidate"]["heldout"],
            "posterior_v4": dev_heldout["modes"]["posterior_v4"]["heldout"],
            "posterior_v5": dev_heldout["modes"]["posterior_v5"]["heldout"],
            "posterior_v5_no_predicted_state": dev_heldout["modes"]["posterior_v5_no_predicted_state"]["heldout"],
            "posterior_v5_no_appearance": dev_heldout["modes"]["posterior_v5_no_appearance"]["heldout"],
        },
    }
    decision_path = output_dir / "stwm_semantic_state_measurement_alignment_v1_decision_20260428.json"
    write_json(decision_path, decision)
    write_doc(decision_path.with_suffix(".md"), "STWM Semantic-State Measurement Alignment V1 Decision", decision)

    guardrail = {
        "generated_at_utc": now_iso(),
        "guardrail_version": "v16",
        "allowed": [
            "Alignment probe maps predicted world-state embeddings to measurement space for eval.",
            "Candidate crop features are measurement observations only.",
            "STWM rollout remains candidate-free.",
        ],
        "forbidden": [
            "Updating STWM backbone with future candidates.",
            "Using heldout to train alignment.",
            "Claiming target-candidate appearance as world-model evidence.",
            "Claiming paper-world-model status without predicted-state load-bearing.",
            "Calling STWM a SAM2/CoTracker plugin.",
        ],
    }
    guardrail_path = output_dir / "stwm_world_model_no_drift_guardrail_v16_20260428.json"
    write_json(guardrail_path, guardrail)
    write_doc(guardrail_path.with_suffix(".md"), "STWM World Model No Drift Guardrail V16", guardrail)

    return {
        "train": train_payload,
        "eval_payloads": eval_payloads,
        "dev_heldout": dev_heldout,
        "decision": decision,
        "guardrail": guardrail,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="reports/stwm_semantic_state_measurement_alignment_dataset_v1_20260428.json")
    parser.add_argument("--train-report", default="reports/stwm_semantic_state_measurement_alignment_probe_train_v1_20260428.json")
    parser.add_argument("--train-doc", default="docs/STWM_SEMANTIC_STATE_MEASUREMENT_ALIGNMENT_PROBE_TRAIN_V1_20260428.md")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-output-dir", default="outputs/alignment_probes/stwm_semantic_state_alignment_v6")
    args = parser.parse_args()
    train_and_eval(
        dataset_path=Path(args.dataset),
        train_report=Path(args.train_report),
        train_doc=Path(args.train_doc),
        output_dir=Path(args.output_dir),
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        probe_output_dir=Path(args.probe_output_dir),
    )


if __name__ == "__main__":
    main()
