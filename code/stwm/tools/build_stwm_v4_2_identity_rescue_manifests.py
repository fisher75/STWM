from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build fixed-ratio manifests for STWM V4.2 identity rescue round")
    parser.add_argument("--base-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument("--base-clip-ids", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json")
    parser.add_argument("--eventful-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_eventful_minival_v1.json")
    parser.add_argument("--hard-query-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_hard_query_minival_v1.json")
    parser.add_argument("--target-size", type=int, default=18)
    parser.add_argument("--eventful-mix-base", type=int, default=6)
    parser.add_argument("--eventful-mix-eventful", type=int, default=12)
    parser.add_argument("--ehq-mix-base", type=int, default=2)
    parser.add_argument("--ehq-mix-eventful", type=int, default=8)
    parser.add_argument("--ehq-mix-hard-query", type=int, default=8)
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/manifests/minisplits")
    parser.add_argument("--name-prefix", default="stwm_v4_2_identity_rescue_v1")
    return parser


def _cycle_take(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or not items:
        return []
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < int(n):
        out.append(items[i % len(items)])
        i += 1
    return out


def _index_by_clip_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(x.get("clip_id", "")): x for x in items if str(x.get("clip_id", ""))}


def _attach_component(item: dict[str, Any], variant: str, component: str, position: int) -> dict[str, Any]:
    out = json.loads(json.dumps(item))
    md = dict(out.get("metadata", {}))
    md["identity_rescue_variant"] = str(variant)
    md["identity_rescue_component"] = str(component)
    md["identity_rescue_position"] = int(position)
    out["metadata"] = md
    return out


def _serialize_variant(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def main() -> None:
    args = build_parser().parse_args()

    base_items = json.loads(Path(args.base_manifest).read_text())
    eventful_items = json.loads(Path(args.eventful_manifest).read_text())
    hard_items = json.loads(Path(args.hard_query_manifest).read_text())

    base_index = _index_by_clip_id(base_items)
    eventful_index = _index_by_clip_id(eventful_items)
    hard_index = _index_by_clip_id(hard_items)

    base_clip_ids = json.loads(Path(args.base_clip_ids).read_text())
    base_pool = [base_index[cid] for cid in base_clip_ids if cid in base_index]
    eventful_pool = [eventful_index[cid] for cid in [x.get("clip_id") for x in eventful_items] if isinstance(cid, str) and cid in eventful_index]
    hard_pool = [hard_index[cid] for cid in [x.get("clip_id") for x in hard_items] if isinstance(cid, str) and cid in hard_index]

    target_size = int(args.target_size)

    control_base_raw = _cycle_take(base_pool, target_size)
    eventful_mix_raw = _cycle_take(base_pool, int(args.eventful_mix_base)) + _cycle_take(eventful_pool, int(args.eventful_mix_eventful))
    ehq_mix_raw = (
        _cycle_take(base_pool, int(args.ehq_mix_base))
        + _cycle_take(eventful_pool, int(args.ehq_mix_eventful))
        + _cycle_take(hard_pool, int(args.ehq_mix_hard_query))
    )

    # Keep exact target size while preserving deterministic ordering.
    eventful_mix_raw = eventful_mix_raw[:target_size]
    ehq_mix_raw = ehq_mix_raw[:target_size]

    control_base = [
        _attach_component(item, "control_resume_base", "base", idx)
        for idx, item in enumerate(control_base_raw)
    ]

    eventful_mix: list[dict[str, Any]] = []
    for idx, item in enumerate(eventful_mix_raw):
        component = "base" if idx < int(args.eventful_mix_base) else "eventful"
        eventful_mix.append(_attach_component(item, "resume_eventful_mix", component, idx))

    ehq_mix: list[dict[str, Any]] = []
    split_a = int(args.ehq_mix_base)
    split_b = int(args.ehq_mix_base) + int(args.ehq_mix_eventful)
    for idx, item in enumerate(ehq_mix_raw):
        if idx < split_a:
            component = "base"
        elif idx < split_b:
            component = "eventful"
        else:
            component = "hard_query"
        ehq_mix.append(_attach_component(item, "resume_eventful_hardquery_mix", component, idx))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    control_path = output_dir / f"{args.name_prefix}_control_resume_base.json"
    eventful_mix_path = output_dir / f"{args.name_prefix}_resume_eventful_mix.json"
    ehq_mix_path = output_dir / f"{args.name_prefix}_resume_eventful_hardquery_mix.json"
    report_path = output_dir / f"{args.name_prefix}_manifest_report.json"

    _serialize_variant(control_path, control_base)
    _serialize_variant(eventful_mix_path, eventful_mix)
    _serialize_variant(ehq_mix_path, ehq_mix)

    def summarize(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        components: dict[str, int] = {}
        clip_ids: list[str] = []
        for it in items:
            md = dict(it.get("metadata", {}))
            comp = str(md.get("identity_rescue_component", "unknown"))
            components[comp] = int(components.get(comp, 0) + 1)
            cid = str(it.get("clip_id", ""))
            if cid:
                clip_ids.append(cid)
        return {
            "variant": name,
            "count": len(items),
            "unique_clip_count": len(set(clip_ids)),
            "component_counts": components,
            "first_clip_ids": clip_ids[:10],
        }

    report = {
        "target_size": target_size,
        "inputs": {
            "base_manifest": str(args.base_manifest),
            "base_clip_ids": str(args.base_clip_ids),
            "eventful_manifest": str(args.eventful_manifest),
            "hard_query_manifest": str(args.hard_query_manifest),
        },
        "outputs": {
            "control_resume_base": str(control_path),
            "resume_eventful_mix": str(eventful_mix_path),
            "resume_eventful_hardquery_mix": str(ehq_mix_path),
            "report": str(report_path),
        },
        "sampling_ratios": {
            "control_resume_base": {"base": int(target_size)},
            "resume_eventful_mix": {
                "base": int(args.eventful_mix_base),
                "eventful": int(args.eventful_mix_eventful),
            },
            "resume_eventful_hardquery_mix": {
                "base": int(args.ehq_mix_base),
                "eventful": int(args.ehq_mix_eventful),
                "hard_query": int(args.ehq_mix_hard_query),
            },
        },
        "summaries": [
            summarize("control_resume_base", control_base),
            summarize("resume_eventful_mix", eventful_mix),
            summarize("resume_eventful_hardquery_mix", ehq_mix),
        ],
    }

    report_path.write_text(json.dumps(report, indent=2))

    print(
        json.dumps(
            {
                "control_manifest": str(control_path),
                "eventful_mix_manifest": str(eventful_mix_path),
                "eventful_hardquery_mix_manifest": str(ehq_mix_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
