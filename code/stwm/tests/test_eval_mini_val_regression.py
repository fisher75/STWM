from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "code" / "stwm" / "evaluators" / "eval_mini_val.py"
DATA_ROOT = REPO_ROOT / "data" / "external"
MANIFEST_TINY = REPO_ROOT / "manifests" / "minisplits" / "vspw_mini.json"
LEGACY_CHECKPOINT = REPO_ROOT / "outputs" / "training" / "week2_minival_v2_2" / "seed_42" / "full" / "checkpoints" / "step_00020.pt"
V42_CHECKPOINT = REPO_ROOT / "outputs" / "training" / "stwm_v4_2_real_220m" / "seed_42" / "full_v4_2" / "checkpoints" / "best.pt"
LEGACY_PRESET_FILE = REPO_ROOT / "code" / "stwm" / "configs" / "model_presets.json"
V42_PRESET_FILE = REPO_ROOT / "code" / "stwm" / "configs" / "model_presets_v4_2.json"

EXPECTED_STABLE_METRICS = {
    "query_localization_error",
    "query_top1_acc",
    "query_hit_rate",
    "identity_consistency",
    "identity_switch_rate",
    "occlusion_recovery_acc",
    "future_trajectory_l1",
    "future_mask_iou",
    "visibility_accuracy",
    "visibility_f1",
}


@pytest.fixture(scope="module", autouse=True)
def _check_prerequisites() -> None:
    required = [
        EVAL_SCRIPT,
        DATA_ROOT,
        MANIFEST_TINY,
        LEGACY_CHECKPOINT,
        V42_CHECKPOINT,
        LEGACY_PRESET_FILE,
        V42_PRESET_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("Missing regression prerequisites: " + ", ".join(missing), allow_module_level=True)


def _run_eval(
    output_path: Path,
    *,
    checkpoint: Path,
    model_preset: str,
    preset_file: Path,
    max_clips: int,
    run_name: str,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "code")

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "stwm",
        "python",
        str(EVAL_SCRIPT),
        "--data-root",
        str(DATA_ROOT),
        "--manifest",
        str(MANIFEST_TINY),
        "--dataset",
        "vspw",
        "--max-clips",
        str(int(max_clips)),
        "--obs-steps",
        "8",
        "--pred-steps",
        "8",
        "--seed",
        "42",
        "--checkpoint",
        str(checkpoint),
        "--model-preset",
        model_preset,
        "--preset-file",
        str(preset_file),
        "--protocol-version",
        "v2_4_detached_frozen",
        "--run-name",
        run_name,
        "--output",
        str(output_path),
        "--device",
        "cpu",
    ]

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return json.loads(output_path.read_text())


def test_legacy_checkpoint_smoke() -> None:
    output = Path("/tmp/stwm_eval_regression_test_legacy_smoke.json")
    result = _run_eval(
        output,
        checkpoint=LEGACY_CHECKPOINT,
        model_preset="prototype_220m",
        preset_file=LEGACY_PRESET_FILE,
        max_clips=1,
        run_name="test_legacy_smoke",
    )

    assert result["num_clips"] == 1
    assert result["model_config"]["family"] == "stwm_1b"
    assert result["protocol"]["requested_protocol_version"] == "v2_4_detached_frozen"
    assert result["protocol"]["protocol_version"] == "v2_3"
    assert result["evaluator_version"] == "v2_4_detached_frozen"


def test_v42_real_checkpoint_smoke() -> None:
    output = Path("/tmp/stwm_eval_regression_test_v42_smoke.json")
    result = _run_eval(
        output,
        checkpoint=V42_CHECKPOINT,
        model_preset="prototype_220m_v4_2",
        preset_file=V42_PRESET_FILE,
        max_clips=1,
        run_name="test_v42_smoke",
    )

    assert result["num_clips"] == 1
    assert result["model_config"]["family"] == "stwm_v4_2"
    assert result["protocol"]["requested_protocol_version"] == "v2_4_detached_frozen"
    assert result["protocol"]["protocol_version"] == "v2_3"
    assert result["evaluator_version"] == "v2_4_detached_frozen"


def test_tiny_manifest_deterministic_regression_schema() -> None:
    output = Path("/tmp/stwm_eval_regression_test_tiny_schema.json")
    result = _run_eval(
        output,
        checkpoint=LEGACY_CHECKPOINT,
        model_preset="prototype_220m",
        preset_file=LEGACY_PRESET_FILE,
        max_clips=2,
        run_name="test_tiny_schema",
    )

    assert result["num_clips"] == 2
    assert set(result["metrics"].keys()) == EXPECTED_STABLE_METRICS
    assert set(result["protocol"]["stable_comparable_metrics"]) == EXPECTED_STABLE_METRICS


def test_same_checkpoint_twice_identical_within_tolerance() -> None:
    out1 = Path("/tmp/stwm_eval_regression_test_determinism_1.json")
    out2 = Path("/tmp/stwm_eval_regression_test_determinism_2.json")

    r1 = _run_eval(
        out1,
        checkpoint=LEGACY_CHECKPOINT,
        model_preset="prototype_220m",
        preset_file=LEGACY_PRESET_FILE,
        max_clips=2,
        run_name="test_det_once",
    )
    r2 = _run_eval(
        out2,
        checkpoint=LEGACY_CHECKPOINT,
        model_preset="prototype_220m",
        preset_file=LEGACY_PRESET_FILE,
        max_clips=2,
        run_name="test_det_once",
    )

    tol = 1e-12
    for key in EXPECTED_STABLE_METRICS:
        diff = abs(float(r1["metrics"][key]) - float(r2["metrics"][key]))
        assert diff <= tol, f"metric {key} diff={diff} > {tol}"
