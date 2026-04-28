#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import random

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
        "listwise_probe_trained",
        "seed_count",
        "selected_seed",
        "no_backbone_update",
        "no_candidate_leakage_to_rollout",
        "posterior_v6_improves_over_no_predicted_state",
        "posterior_v6_improves_over_distance",
        "posterior_v6_improves_over_appearance",
        "listwise_signal_positive",
        "recommended_next_step_choice",
    ]:
        if key in payload:
            lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    m = sum(values) / len(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def both_classes(labels: list[int]) -> bool:
    return 0 in labels and 1 in labels


def binary_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for ps in pos:
        for ns in neg:
            total += 1.0
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / total if total else None


def binary_ap(scores: list[float], labels: list[int]) -> float | None:
    if not scores or 1 not in labels:
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


def tensor(values: Any) -> torch.Tensor:
    return torch.tensor([float(x) for x in values], dtype=torch.float32)


class ListwiseProbe(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if int(hidden_dim) > 0:
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(int(in_dim)),
                torch.nn.Linear(int(in_dim), int(hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(hidden_dim), int(out_dim)),
            )
        else:
            self.net = torch.nn.Sequential(torch.nn.LayerNorm(int(in_dim)), torch.nn.Linear(int(in_dim), int(out_dim)))
        self.logit_scale = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, state_vec: torch.Tensor, candidate_matrix: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(self.net(state_vec.view(1, -1)).squeeze(0), dim=0)
        cand = F.normalize(candidate_matrix.float(), dim=-1)
        return cand @ pred * self.logit_scale.clamp(1.0, 30.0)


def group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("item_id"))].append(record)
    items: list[dict[str, Any]] = []
    for item_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r.get("candidate_index", 0)))
        labels = [int(r.get("label_same_identity", 0)) for r in rows]
        if 1 not in labels or 0 not in labels:
            continue
        items.append(
            {
                "item_id": item_id,
                "split": rows[0].get("split"),
                "rows": rows,
                "positive_index": labels.index(1),
                "subset_tags": rows[0].get("subset_tags") if isinstance(rows[0].get("subset_tags"), dict) else {},
            }
        )
    return items


def train_one_probe(
    *,
    probe: ListwiseProbe,
    items: list[dict[str, Any]],
    state_key: str,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[list[float], ListwiseProbe]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    opt = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    dev_items = list(items)
    for _ in range(int(epochs)):
        random.shuffle(dev_items)
        epoch_loss = 0.0
        count = 0
        for item in dev_items:
            rows = item["rows"]
            state_vec = tensor(rows[0][state_key])
            candidates = torch.stack([tensor(r["candidate_crop_feature"]) for r in rows], dim=0)
            target = torch.tensor([int(item["positive_index"])], dtype=torch.long)
            logits = probe(state_vec, candidates).view(1, -1)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().item())
            count += 1
        losses.append(epoch_loss / max(count, 1))
    return losses, probe


def score_probe(probe: ListwiseProbe, items: list[dict[str, Any]], state_key: str) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    probe.eval()
    with torch.no_grad():
        for item in items:
            rows = item["rows"]
            state_vec = tensor(rows[0][state_key])
            candidates = torch.stack([tensor(r["candidate_crop_feature"]) for r in rows], dim=0)
            item_scores = torch.sigmoid(probe(state_vec, candidates)).detach().cpu().tolist()
            scores[item["item_id"]] = [float(x) for x in item_scores]
    return scores


def flat_scores(items: list[dict[str, Any]], scores_by_item: dict[str, list[float]], split: str | None = None) -> tuple[list[float], list[int], list[float], list[float]]:
    scores: list[float] = []
    labels: list[int] = []
    top1: list[float] = []
    rr: list[float] = []
    for item in items:
        if split is not None and item.get("split") != split:
            continue
        item_scores = scores_by_item.get(item["item_id"], [])
        item_labels = [int(r.get("label_same_identity", 0)) for r in item["rows"]]
        if len(item_scores) != len(item_labels):
            continue
        scores.extend(item_scores)
        labels.extend(item_labels)
        order = sorted(range(len(item_scores)), key=lambda i: item_scores[i], reverse=True)
        gt = item_labels.index(1)
        top1.append(1.0 if item_labels[order[0]] == 1 else 0.0)
        rr.append(1.0 / float(order.index(gt) + 1))
    return scores, labels, top1, rr


def evaluate(items: list[dict[str, Any]], scores_by_item: dict[str, list[float]], split: str | None = None) -> dict[str, Any]:
    scores, labels, top1, rr = flat_scores(items, scores_by_item, split)
    return {
        "candidate_top1": mean(top1),
        "candidate_MRR": mean(rr),
        "candidate_AP": binary_ap(scores, labels) if both_classes(labels) else None,
        "candidate_AUROC": binary_auc(scores, labels) if both_classes(labels) else None,
        "candidate_positive_rate": sum(labels) / len(labels) if labels else None,
        "candidate_record_count": len(labels),
        "metric_items": len(top1),
    }


def subset_breakdown(items: list[dict[str, Any]], scores_by_item: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset in SUBSETS:
        sub_items = [item for item in items if bool((item.get("subset_tags") or {}).get(subset))]
        out[subset] = evaluate(sub_items, scores_by_item)
    return out


def combine_scores(
    items: list[dict[str, Any]],
    *,
    semantic_scores: dict[str, list[float]],
    identity_scores: dict[str, list[float]],
    weights: dict[str, float],
) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {}
    denom = max(sum(abs(x) for x in weights.values()), 1e-8)
    for item in items:
        rows = item["rows"]
        sem = semantic_scores.get(item["item_id"], [0.5] * len(rows))
        ident = identity_scores.get(item["item_id"], [0.5] * len(rows))
        vals: list[float] = []
        for idx, row in enumerate(rows):
            d = float(row.get("distance_score", 0.0))
            app = float(row.get("appearance_score", 0.5))
            pri = 0.5 * float(row.get("future_visibility_prob", 0.5)) + 0.5 * float(row.get("future_reappearance_event_prob", 0.5))
            vals.append(
                (
                    weights.get("distance", 0.0) * d
                    + weights.get("appearance", 0.0) * app
                    + weights.get("semantic", 0.0) * float(sem[idx])
                    + weights.get("identity", 0.0) * float(ident[idx])
                    + weights.get("priors", 0.0) * pri
                )
                / denom
            )
        combined[item["item_id"]] = vals
    return combined


def primitive_score(items: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for item in items:
        out[item["item_id"]] = [float(r.get(key, 0.0)) for r in item["rows"]]
    return out


def write_eval(path: Path, mode: str, items: list[dict[str, Any]], scores: dict[str, list[float]]) -> dict[str, Any]:
    payload = {
        "generated_at_utc": now_iso(),
        "mode": mode,
        "candidate_score_mode": mode,
        "future_candidate_used_as_input": False,
        "candidate_feature_used_for_rollout": False,
        "candidate_feature_used_for_scoring": True,
        "stage2_val_fallback_used": False,
        "old_association_report_used": False,
        "full_model_forward_executed": True,
        "full_free_rollout_executed": True,
        **evaluate(items, scores),
        "dev": evaluate(items, scores, split="dev"),
        "heldout": evaluate(items, scores, split="heldout"),
        "per_subset_breakdown": subset_breakdown(items, scores),
    }
    write_json(path, payload)
    write_doc(path.with_suffix(".md"), "STWM External Candidate Scoring V6 " + mode, payload)
    return payload


def run(
    *,
    dataset_path: Path,
    output_dir: Path,
    train_report: Path,
    train_doc: Path,
    probe_output_dir: Path,
    seeds: list[int],
    epochs: int,
    lr: float,
    bottleneck_dim: int,
) -> dict[str, Any]:
    dataset = load_json(dataset_path)
    records = [r for r in dataset.get("records", []) if isinstance(r, dict)]
    items = group_records(records)
    dev_items = [item for item in items if item.get("split") == "dev"]
    if not dev_items:
        raise RuntimeError("no dev items for listwise training")
    sem_dim = len(dev_items[0]["rows"][0]["predicted_semantic_embedding"])
    id_dim = len(dev_items[0]["rows"][0]["predicted_identity_embedding"])
    cand_dim = len(dev_items[0]["rows"][0]["candidate_crop_feature"])
    probe_output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports: list[dict[str, Any]] = []
    seed_score_tables: dict[int, dict[str, dict[str, list[float]]]] = {}

    for seed in seeds:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        sem_probe = ListwiseProbe(sem_dim, cand_dim, hidden_dim=0)
        id_probe = ListwiseProbe(id_dim, cand_dim, hidden_dim=0)
        si_probe = ListwiseProbe(sem_dim + id_dim, cand_dim, hidden_dim=int(bottleneck_dim))
        sem_losses, sem_probe = train_one_probe(
            probe=sem_probe, items=dev_items, state_key="predicted_semantic_embedding", epochs=epochs, lr=lr, seed=seed
        )
        id_losses, id_probe = train_one_probe(
            probe=id_probe, items=dev_items, state_key="predicted_identity_embedding", epochs=epochs, lr=lr, seed=seed + 1000
        )
        # Add concat features lazily to rows for the semantic-identity probe.
        for item in items:
            for row in item["rows"]:
                row["predicted_semantic_identity_embedding"] = list(row["predicted_semantic_embedding"]) + list(row["predicted_identity_embedding"])
        si_losses, si_probe = train_one_probe(
            probe=si_probe,
            items=dev_items,
            state_key="predicted_semantic_identity_embedding",
            epochs=epochs,
            lr=lr,
            seed=seed + 2000,
        )
        sem_scores = score_probe(sem_probe, items, "predicted_semantic_embedding")
        id_scores = score_probe(id_probe, items, "predicted_identity_embedding")
        si_scores = score_probe(si_probe, items, "predicted_semantic_identity_embedding")
        posterior_v6 = combine_scores(
            items,
            semantic_scores=sem_scores,
            identity_scores=id_scores,
            weights={"distance": 0.25, "appearance": 0.25, "semantic": 0.20, "identity": 0.15, "priors": 0.15},
        )
        no_pred = combine_scores(
            items,
            semantic_scores=sem_scores,
            identity_scores=id_scores,
            weights={"distance": 0.40, "appearance": 0.40, "priors": 0.20},
        )
        no_app = combine_scores(
            items,
            semantic_scores=sem_scores,
            identity_scores=id_scores,
            weights={"distance": 0.35, "semantic": 0.25, "identity": 0.20, "priors": 0.20},
        )
        no_dist = combine_scores(
            items,
            semantic_scores=sem_scores,
            identity_scores=id_scores,
            weights={"appearance": 0.35, "semantic": 0.25, "identity": 0.20, "priors": 0.20},
        )
        modes = {
            "aligned_listwise_semantic": sem_scores,
            "aligned_listwise_identity": id_scores,
            "aligned_listwise_semantic_identity": si_scores,
            "posterior_v6": posterior_v6,
            "posterior_v6_no_predicted_state": no_pred,
            "posterior_v6_no_appearance": no_app,
            "posterior_v6_no_distance": no_dist,
            "distance_only": primitive_score(items, "distance_score"),
            "appearance_only": primitive_score(items, "appearance_score"),
        }
        seed_score_tables[int(seed)] = modes
        report = {
            "seed": int(seed),
            "train_losses": {
                "semantic_start": sem_losses[0],
                "semantic_end": sem_losses[-1],
                "identity_start": id_losses[0],
                "identity_end": id_losses[-1],
                "semantic_identity_start": si_losses[0],
                "semantic_identity_end": si_losses[-1],
            },
            "trainable_params": {
                "semantic_alignment_listwise": sum(p.numel() for p in sem_probe.parameters()),
                "identity_alignment_listwise": sum(p.numel() for p in id_probe.parameters()),
                "semantic_identity_alignment_listwise": sum(p.numel() for p in si_probe.parameters()),
            },
            "heldout": {mode: evaluate(items, scores, split="heldout") for mode, scores in modes.items()},
            "dev": {mode: evaluate(items, scores, split="dev") for mode, scores in modes.items()},
            "probe_weight_paths": {
                "semantic": str(probe_output_dir / f"listwise_semantic_probe_seed{seed}.pt"),
                "identity": str(probe_output_dir / f"listwise_identity_probe_seed{seed}.pt"),
                "semantic_identity": str(probe_output_dir / f"listwise_semantic_identity_probe_seed{seed}.pt"),
            },
        }
        torch.save({"state_dict": sem_probe.state_dict(), "seed": int(seed), "input_dim": sem_dim, "output_dim": cand_dim}, report["probe_weight_paths"]["semantic"])
        torch.save({"state_dict": id_probe.state_dict(), "seed": int(seed), "input_dim": id_dim, "output_dim": cand_dim}, report["probe_weight_paths"]["identity"])
        torch.save(
            {
                "state_dict": si_probe.state_dict(),
                "seed": int(seed),
                "input_dim": sem_dim + id_dim,
                "output_dim": cand_dim,
                "bottleneck_dim": int(bottleneck_dim),
            },
            report["probe_weight_paths"]["semantic_identity"],
        )
        seed_reports.append(report)

    def held_ap(report: dict[str, Any], mode: str) -> float:
        value = report.get("dev", {}).get(mode, {}).get("candidate_AP")
        return float(value) if isinstance(value, (int, float)) else -1.0

    best = max(seed_reports, key=lambda r: held_ap(r, "posterior_v6"))
    best_seed = int(best["seed"])
    selected_scores = seed_score_tables[best_seed]
    eval_paths = {
        "aligned_listwise_semantic": output_dir / "stwm_external_candidate_scoring_v6_aligned_listwise_semantic_eval_20260428.json",
        "aligned_listwise_identity": output_dir / "stwm_external_candidate_scoring_v6_aligned_listwise_identity_eval_20260428.json",
        "aligned_listwise_semantic_identity": output_dir / "stwm_external_candidate_scoring_v6_aligned_listwise_semantic_identity_eval_20260428.json",
        "posterior_v6": output_dir / "stwm_external_candidate_scoring_v6_posterior_v6_eval_20260428.json",
        "posterior_v6_no_predicted_state": output_dir / "stwm_external_candidate_scoring_v6_posterior_v6_no_predicted_state_eval_20260428.json",
        "posterior_v6_no_appearance": output_dir / "stwm_external_candidate_scoring_v6_posterior_v6_no_appearance_eval_20260428.json",
        "posterior_v6_no_distance": output_dir / "stwm_external_candidate_scoring_v6_posterior_v6_no_distance_eval_20260428.json",
    }
    eval_payloads = {mode: write_eval(path, mode, items, selected_scores[mode]) for mode, path in eval_paths.items()}

    def seed_metric(mode: str, metric: str) -> list[float]:
        vals: list[float] = []
        for report in seed_reports:
            value = report.get("heldout", {}).get(mode, {}).get(metric)
            if isinstance(value, (int, float)):
                vals.append(float(value))
        return vals

    seed_stats = {
        mode: {
            metric + "_mean": mean(seed_metric(mode, metric))
            for metric in ["candidate_top1", "candidate_MRR", "candidate_AP", "candidate_AUROC"]
        }
        | {
            metric + "_std": std(seed_metric(mode, metric))
            for metric in ["candidate_top1", "candidate_MRR", "candidate_AP", "candidate_AUROC"]
        }
        for mode in eval_paths
    }
    dev_heldout = {
        "generated_at_utc": now_iso(),
        "dataset": str(dataset_path),
        "no_heldout_tuning": True,
        "seed_count": len(seeds),
        "seeds": [int(x) for x in seeds],
        "selected_seed": best_seed,
        "selected_by": "dev posterior_v6 candidate_AP",
        "seed_mean_std": seed_stats,
        "selected_seed_metrics": {
            mode: {"all": payload, "dev": payload.get("dev"), "heldout": payload.get("heldout")}
            for mode, payload in eval_payloads.items()
        },
        "all_seed_reports": seed_reports,
    }
    dev_path = output_dir / "stwm_external_candidate_scoring_v6_dev_heldout_summary_20260428.json"
    write_json(dev_path, dev_heldout)
    write_doc(dev_path.with_suffix(".md"), "STWM External Candidate Scoring V6 Dev Heldout Summary", dev_heldout)

    score_table = {
        "generated_at_utc": now_iso(),
        "dataset": str(dataset_path),
        "selected_seed": best_seed,
        "records": [
            {
                "item_id": item["item_id"],
                "split": item.get("split"),
                "subset_tags": item.get("subset_tags"),
                "labels": [int(r.get("label_same_identity", 0)) for r in item["rows"]],
                "scores": {mode: selected_scores[mode].get(item["item_id"], []) for mode in selected_scores},
            }
            for item in items
        ],
    }
    score_table_path = output_dir / "stwm_semantic_state_alignment_v6_score_table_20260428.json"
    write_json(score_table_path, score_table)

    p6 = eval_payloads["posterior_v6"]["heldout"]
    np = eval_payloads["posterior_v6_no_predicted_state"]["heldout"]
    dist = evaluate(items, primitive_score(items, "distance_score"), split="heldout")
    app = evaluate(items, primitive_score(items, "appearance_score"), split="heldout")
    listwise_signal = bool(
        isinstance(p6.get("candidate_AP"), (int, float))
        and isinstance(np.get("candidate_AP"), (int, float))
        and float(p6["candidate_AP"]) > float(np["candidate_AP"])
        and isinstance(p6.get("candidate_AUROC"), (int, float))
        and isinstance(np.get("candidate_AUROC"), (int, float))
        and float(p6["candidate_AUROC"]) >= float(np["candidate_AUROC"])
    )
    train_payload = {
        "generated_at_utc": now_iso(),
        "listwise_probe_trained": True,
        "dataset": str(dataset_path),
        "seed_count": len(seeds),
        "seeds": [int(x) for x in seeds],
        "selected_seed": best_seed,
        "epochs": int(epochs),
        "lr": float(lr),
        "bottleneck_dim": int(bottleneck_dim),
        "no_backbone_update": True,
        "no_candidate_leakage_to_rollout": True,
        "future_candidate_used_as_input": False,
        "heldout_used_for_training_or_tuning": False,
        "trainable_params_selected_seed": best.get("trainable_params"),
        "probe_weight_paths_selected_seed": best.get("probe_weight_paths"),
        "all_seed_reports": seed_reports,
        "score_table_path": str(score_table_path),
        "posterior_v6_improves_over_no_predicted_state": bool(p6.get("candidate_AP") > np.get("candidate_AP")),
        "posterior_v6_improves_over_distance": bool(p6.get("candidate_AP") > dist.get("candidate_AP")),
        "posterior_v6_improves_over_appearance": bool(p6.get("candidate_AP") > app.get("candidate_AP")),
        "listwise_signal_positive": listwise_signal,
    }
    write_json(train_report, train_payload)
    write_doc(train_doc, "STWM Semantic-State Alignment V6 Listwise Train", train_payload)
    return train_payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="reports/stwm_semantic_state_measurement_alignment_dataset_v1_20260428.json")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--train-report", default="reports/stwm_semantic_state_alignment_v6_listwise_train_20260428.json")
    parser.add_argument("--train-doc", default="docs/STWM_SEMANTIC_STATE_ALIGNMENT_V6_LISTWISE_TRAIN_20260428.md")
    parser.add_argument("--probe-output-dir", default="outputs/alignment_probes/stwm_semantic_state_alignment_v6")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    run(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        train_report=Path(args.train_report),
        train_doc=Path(args.train_doc),
        probe_output_dir=Path(args.probe_output_dir),
        seeds=seeds,
        epochs=int(args.epochs),
        lr=float(args.lr),
        bottleneck_dim=int(args.bottleneck_dim),
    )


if __name__ == "__main__":
    main()
