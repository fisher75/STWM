# STWM External Baseline Smoke Decision 20260426

```json
{
  "cotracker_enter_full_eval": true,
  "created_at": "2026-04-27T15:34:52+0800",
  "cutie_enter_full_eval": true,
  "decision_rules": {
    "cutie_or_sam2_pass_does_not_wait_for_cotracker": true,
    "minimum_successful_items_for_pass": 5
  },
  "needs_adapter_fix": {
    "cotracker": false,
    "cutie": false,
    "sam2": false
  },
  "next_step_choice": "run_external_baseline_full_eval_next",
  "priority_full_eval_baseline": "cotracker",
  "sam2_enter_full_eval": true,
  "summaries": {
    "cotracker": {
      "attempted_items": 10,
      "average_runtime": 0.50328,
      "baseline_name": "cotracker",
      "common_failure_reasons": {},
      "exact_blocking_reason": null,
      "example_outputs": [
        {
          "baseline_name": "cotracker",
          "candidate_point_inside_ratio": {
            "1": 0.0009662336578248841,
            "2": 0.26024434077090497,
            "3": 0.6306283357805812
          },
          "candidate_scores": {
            "1": 0.0009662336578248841,
            "2": 0.26024434077090497,
            "3": 0.6306283357805812
          },
          "centroid_tiebreak_score": "0.001 * inverse normalized centroid distance",
          "checkpoint_used": "baselines/checkpoints/cotracker/scaled_offline.pth",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "3",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3",
          "mrr": 1.0,
          "predicted_candidate_id": "3",
          "protocol_item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3",
          "runtime_seconds": 0.5476,
          "sampled_point_count": 33,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "crossing_ambiguity": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "3",
            "2",
            "1"
          ],
          "visible_point_count": 27,
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cotracker",
          "candidate_point_inside_ratio": {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 0.0,
            "5": 0.0,
            "6": 0.0
          },
          "candidate_scores": {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 0.0,
            "5": 0.0,
            "6": 0.0
          },
          "centroid_tiebreak_score": "0.001 * inverse normalized centroid distance",
          "checkpoint_used": "baselines/checkpoints/cotracker/scaled_offline.pth",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "2",
          "gt_rank": 2,
          "item_id": "burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2",
          "mrr": 0.5,
          "predicted_candidate_id": "1",
          "protocol_item_id": "burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2",
          "runtime_seconds": 0.5884,
          "sampled_point_count": 23,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "legacy_85_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "appearance_change": true,
            "crossing_ambiguity": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "1",
            "2",
            "3",
            "4",
            "5"
          ],
          "visible_point_count": 0,
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cotracker",
          "candidate_point_inside_ratio": {
            "10": 0.0006273036739325542,
            "2": 0.0005598610231664332,
            "3": 0.000766302393273769,
            "9": 0.0006557514809546239
          },
          "candidate_scores": {
            "10": 0.0006273036739325542,
            "2": 0.0005598610231664332,
            "3": 0.000766302393273769,
            "9": 0.0006557514809546239
          },
          "centroid_tiebreak_score": "0.001 * inverse normalized centroid distance",
          "checkpoint_used": "baselines/checkpoints/cotracker/scaled_offline.pth",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "2",
          "gt_rank": 4,
          "item_id": "burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2",
          "mrr": 0.25,
          "predicted_candidate_id": "3",
          "protocol_item_id": "burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2",
          "runtime_seconds": 0.5002,
          "sampled_point_count": 64,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "appearance_change": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "3",
            "9",
            "10",
            "2"
          ],
          "visible_point_count": 60,
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cotracker",
          "candidate_point_inside_ratio": {
            "3": 0.0008731759731364883,
            "4": 0.000961492756454589,
            "5": 1.0009951910684802,
            "7": 0.0008141829561588758
          },
          "candidate_scores": {
            "3": 0.0008731759731364883,
            "4": 0.000961492756454589,
            "5": 1.0009951910684802,
            "7": 0.0008141829561588758
          },
          "centroid_tiebreak_score": "0.001 * inverse normalized centroid distance",
          "checkpoint_used": "baselines/checkpoints/cotracker/scaled_offline.pth",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "5",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5",
          "mrr": 1.0,
          "predicted_candidate_id": "5",
          "protocol_item_id": "burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5",
          "runtime_seconds": 0.4527,
          "sampled_point_count": 64,
          "source_protocol": [
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving"
          ],
          "subset_tags": {
            "appearance_change": true,
            "crossing_ambiguity": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "5",
            "4",
            "3",
            "7"
          ],
          "visible_point_count": 30,
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cotracker",
          "candidate_point_inside_ratio": {
            "1": 0.9072495741050066,
            "2": 0.04785498495048018,
            "3": 0.0009658569607454805
          },
          "candidate_scores": {
            "1": 0.9072495741050066,
            "2": 0.04785498495048018,
            "3": 0.0009658569607454805
          },
          "centroid_tiebreak_score": "0.001 * inverse normalized centroid distance",
          "checkpoint_used": "baselines/checkpoints/cotracker/scaled_offline.pth",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "1",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1",
          "mrr": 1.0,
          "predicted_candidate_id": "1",
          "protocol_item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1",
          "runtime_seconds": 0.5292,
          "sampled_point_count": 64,
          "source_protocol": [
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving"
          ],
          "subset_tags": {
            "crossing_ambiguity": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "1",
            "2",
            "3"
          ],
          "visible_point_count": 64,
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/smoke_contact_sheet.png"
          }
        }
      ],
      "failed_items": 0,
      "model_load_error": null,
      "mrr_smoke": 0.795,
      "smoke_pass": true,
      "successful_items": 10,
      "top1_smoke": 0.7,
      "visual_output_paths": [
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__21e37598-52d4-345c-8ef9-03ae19615d3d__6/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__21e37598-52d4-345c-8ef9-03ae19615d3d__6/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__4/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cotracker/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__4/overlays/smoke_contact_sheet.png"
      ]
    },
    "cutie": {
      "attempted_items": 10,
      "average_runtime": 1.06453,
      "baseline_name": "cutie",
      "common_failure_reasons": {},
      "exact_blocking_reason": null,
      "example_outputs": [
        {
          "baseline_name": "cutie",
          "candidate_scores": {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0
          },
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "3",
          "gt_rank": 3,
          "item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3",
          "mrr": 0.3333333333333333,
          "output_mask_path": null,
          "predicted_candidate_id": "1",
          "protocol_item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3",
          "runtime_seconds": 3.428,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "crossing_ambiguity": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "1",
            "2",
            "3"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cutie",
          "candidate_scores": {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 0.0,
            "5": 0.0,
            "6": 0.0
          },
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "2",
          "gt_rank": 2,
          "item_id": "burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2",
          "mrr": 0.5,
          "output_mask_path": null,
          "predicted_candidate_id": "1",
          "protocol_item_id": "burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2",
          "runtime_seconds": 0.7669,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "legacy_85_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "appearance_change": true,
            "crossing_ambiguity": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "1",
            "2",
            "3",
            "4",
            "5"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cutie/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cutie/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cutie",
          "candidate_scores": {
            "10": 0.0,
            "2": 0.0,
            "3": 0.0,
            "9": 0.0
          },
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "2",
          "gt_rank": 2,
          "item_id": "burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2",
          "mrr": 0.5,
          "output_mask_path": null,
          "predicted_candidate_id": "10",
          "protocol_item_id": "burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2",
          "runtime_seconds": 0.8586,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "appearance_change": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "10",
            "2",
            "3",
            "9"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cutie/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cutie/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cutie",
          "candidate_scores": {
            "3": 0.0,
            "4": 0.0,
            "5": 0.8962962962962963,
            "7": 0.0
          },
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "5",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5",
          "mrr": 1.0,
          "output_mask_path": null,
          "predicted_candidate_id": "5",
          "protocol_item_id": "burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5",
          "runtime_seconds": 0.8578,
          "source_protocol": [
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving"
          ],
          "subset_tags": {
            "appearance_change": true,
            "crossing_ambiguity": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "5",
            "3",
            "4",
            "7"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cutie/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cutie/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "cutie",
          "candidate_scores": {
            "1": 0.8540145985401459,
            "2": 0.013605442176870748,
            "3": 0.0
          },
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "gt_candidate_id": "1",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1",
          "mrr": 1.0,
          "output_mask_path": null,
          "predicted_candidate_id": "1",
          "protocol_item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1",
          "runtime_seconds": 0.7735,
          "source_protocol": [
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving"
          ],
          "subset_tags": {
            "crossing_ambiguity": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "1",
            "2",
            "3"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/smoke_contact_sheet.png"
          }
        }
      ],
      "failed_items": 0,
      "model_load_error": null,
      "mrr_smoke": 0.65,
      "smoke_pass": true,
      "successful_items": 10,
      "top1_smoke": 0.4,
      "visual_output_paths": [
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__21e37598-52d4-345c-8ef9-03ae19615d3d__6/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__21e37598-52d4-345c-8ef9-03ae19615d3d__6/overlays/smoke_contact_sheet.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__4/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/cutie/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__4/overlays/smoke_contact_sheet.png"
      ]
    },
    "sam2": {
      "attempted_items": 10,
      "average_runtime": 2.05274,
      "baseline_name": "sam2",
      "common_failure_reasons": {},
      "exact_blocking_reason": null,
      "example_outputs": [
        {
          "baseline_name": "sam2",
          "candidate_scores": {
            "1": 0.0,
            "2": 0.017241379310344827,
            "3": 0.56
          },
          "checkpoint_used": "baselines/checkpoints/sam2/sam2.1_hiera_tiny.pt",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "failure_stage_if_any": null,
          "gt_candidate_id": "3",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3",
          "mrr": 1.0,
          "predicted_candidate_id": "3",
          "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
          "prompt_type_used": "box",
          "protocol_item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::3",
          "runtime_seconds": 2.6405,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "crossing_ambiguity": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "3",
            "2",
            "1"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "sam2",
          "candidate_scores": {
            "1": 0.8166750376695128,
            "2": 0.00030432136335970786,
            "3": 0.09624413145539906,
            "4": 0.003501750875437719,
            "5": 0.0,
            "6": 0.003319108582266477
          },
          "checkpoint_used": "baselines/checkpoints/sam2/sam2.1_hiera_tiny.pt",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "failure_stage_if_any": null,
          "gt_candidate_id": "2",
          "gt_rank": 5,
          "item_id": "burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2",
          "mrr": 0.2,
          "predicted_candidate_id": "1",
          "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
          "prompt_type_used": "box",
          "protocol_item_id": "burst::ArgoVerse::rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a::2",
          "runtime_seconds": 2.0805,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "legacy_85_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "appearance_change": true,
            "crossing_ambiguity": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "1",
            "3",
            "4",
            "6",
            "2"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/sam2/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/sam2/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "sam2",
          "candidate_scores": {
            "10": 0.0,
            "2": 0.0,
            "3": 0.0,
            "9": 0.0
          },
          "checkpoint_used": "baselines/checkpoints/sam2/sam2.1_hiera_tiny.pt",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "failure_stage_if_any": null,
          "gt_candidate_id": "2",
          "gt_rank": 2,
          "item_id": "burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2",
          "mrr": 0.5,
          "predicted_candidate_id": "10",
          "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
          "prompt_type_used": "box",
          "protocol_item_id": "burst::ArgoVerse::side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6::2",
          "runtime_seconds": 2.09,
          "source_protocol": [
            "densified_200_context_preserving",
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving",
            "reacquisition_v2"
          ],
          "subset_tags": {
            "appearance_change": true,
            "long_gap_persistence": true,
            "occlusion_reappearance": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": false,
          "top5_candidates": [
            "10",
            "2",
            "3",
            "9"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/sam2/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/sam2/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "sam2",
          "candidate_scores": {
            "3": 0.0,
            "4": 0.0,
            "5": 0.9390243902439024,
            "7": 0.0
          },
          "checkpoint_used": "baselines/checkpoints/sam2/sam2.1_hiera_tiny.pt",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "failure_stage_if_any": null,
          "gt_candidate_id": "5",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5",
          "mrr": 1.0,
          "predicted_candidate_id": "5",
          "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
          "prompt_type_used": "box",
          "protocol_item_id": "burst::ArgoVerse::043aeba7-14e5-3cde-8a5c-639389b6d3a6::5",
          "runtime_seconds": 2.1068,
          "source_protocol": [
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving"
          ],
          "subset_tags": {
            "appearance_change": true,
            "crossing_ambiguity": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "5",
            "3",
            "4",
            "7"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/sam2/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/sam2/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/smoke_contact_sheet.png"
          }
        },
        {
          "baseline_name": "sam2",
          "candidate_scores": {
            "1": 0.8006756756756757,
            "2": 0.09491525423728814,
            "3": 0.006644518272425249
          },
          "checkpoint_used": "baselines/checkpoints/sam2/sam2.1_hiera_tiny.pt",
          "dataset": "BURST",
          "failure_reason_if_any": null,
          "failure_stage_if_any": null,
          "gt_candidate_id": "1",
          "gt_rank": 1,
          "item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1",
          "mrr": 1.0,
          "predicted_candidate_id": "1",
          "predictor_class_or_api": "sam2.sam2_video_predictor.SAM2VideoPredictor",
          "prompt_type_used": "box",
          "protocol_item_id": "burst::ArgoVerse::10b8dee6-778f-33e4-a946-d842d2d9c3d7::1",
          "runtime_seconds": 2.0801,
          "source_protocol": [
            "heldout_burst_heavy_context_preserving",
            "protocol_v3_extended_600_context_preserving"
          ],
          "subset_tags": {
            "crossing_ambiguity": true,
            "small_object": true
          },
          "success": true,
          "top1_correct": true,
          "top5_candidates": [
            "1",
            "2",
            "3"
          ],
          "visual_overlays": {
            "predicted_vs_gt_overlay": "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/predicted_vs_gt_overlay.png",
            "smoke_contact_sheet": "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/smoke_contact_sheet.png"
          }
        }
      ],
      "failed_items": 0,
      "model_load_error": null,
      "mrr_smoke": 0.7033333333333334,
      "smoke_pass": true,
      "successful_items": 10,
      "top1_smoke": 0.5,
      "visual_output_paths": [
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__rear_right_02cf0ce1-699a-373b-86c0-eb6fd5f4697a__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__side_left_043aeba7-14e5-3cde-8a5c-639389b6d3a6__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__5/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__1/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__2/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__10b8dee6-778f-33e4-a946-d842d2d9c3d7__2/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__043aeba7-14e5-3cde-8a5c-639389b6d3a6__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__3/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__3/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__21e37598-52d4-345c-8ef9-03ae19615d3d__6/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__21e37598-52d4-345c-8ef9-03ae19615d3d__6/overlays/smoke_contact_sheet.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__4/overlays/predicted_vs_gt_overlay.png",
        "baselines/outputs/sam2/smoke/burst__ArgoVerse__45753856-4575-4575-4575-345754906624__4/overlays/smoke_contact_sheet.png"
      ]
    }
  }
}
```
