#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_HF_GGUFS = {
    "mistral-7b-instruct-v0.3": (
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    ),
    "qwen2.5-3b-instruct": (
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
    "qwen2.5-3b": (
        "Qwen/Qwen2.5-3B-GGUF",
        "qwen2.5-3b-q4_k_m.gguf",
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    log("$ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def run_capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, cwd=str(cwd) if cwd else None)
    return result.stdout.strip()


def bool_arg(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def maybe_install_deps(root: Path, cuda: bool) -> None:
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    if cuda:
        env = dict(os.environ)
        env.setdefault("CMAKE_ARGS", "-DGGML_CUDA=on")
        env.setdefault("FORCE_CMAKE", "1")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-cache-dir",
                "llama-cpp-python>=0.2.90",
            ],
            env=env,
        )
    else:
        run([sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.90"])
    run([sys.executable, "-m", "pip", "install", "-e", ".[data]", "huggingface_hub"], cwd=root)


def parse_hf_model(values: list[str]) -> dict[str, tuple[str, str]]:
    mapping = dict(DEFAULT_HF_GGUFS)
    for value in values:
        try:
            model_id, spec = value.split("=", 1)
            repo_id, filename = spec.split(":", 1)
        except ValueError as exc:
            raise SystemExit(
                "--hf-model must look like model_id=repo_id:filename, "
                f"got: {value!r}"
            ) from exc
        mapping[model_id.strip()] = (repo_id.strip(), filename.strip())
    return mapping


def download_model_paths(model_ids: list[str], out_dir: Path, hf_models: dict[str, tuple[str, str]]) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    model_paths: dict[str, str] = {}
    for model_id in model_ids:
        if model_id not in hf_models:
            known = ", ".join(sorted(hf_models))
            raise SystemExit(f"No HF GGUF mapping for {model_id!r}. Known: {known}. Use --hf-model to add one.")
        repo_id, filename = hf_models[model_id]
        log(f"Downloading {model_id}: {repo_id}/{filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(out_dir))
        model_paths[model_id] = str(Path(path).resolve())
    return model_paths


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run Phase 4 decoding sampler confirmation on Colab/A100 without a notebook. "
            "If the GitHub repo is private, clone it in Colab using a PAT (e.g. Colab secret GITHUB_TOKEN) "
            "before invoking this script."
        )
    )
    p.add_argument("--root", default=".", help="Repository root in Colab.")
    p.add_argument("--output-root", default="outputs/phase4_colab_a100_mistral_v1")
    p.add_argument("--backend", default="llama_cpp")
    p.add_argument("--models", default="mistral-7b-instruct-v0.3")
    p.add_argument("--sampler-profiles", default="default_nucleus,anti_repetition,spread_topk_minp")
    p.add_argument("--tasks", default="cdat,aut")
    p.add_argument("--methods", default="one_shot")
    p.add_argument("--temperatures", default="0.7")
    p.add_argument("--seeds", default="11,37,73")
    p.add_argument("--limit-cues", type=int, default=8)
    p.add_argument("--limit-aut", type=int, default=8)
    p.add_argument("--cue-offset", type=int, default=0)
    p.add_argument("--aut-offset", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=224)
    p.add_argument("--token-budget-per-prompt", type=int, default=224)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--n-ctx", type=int, default=16384)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--n-batch", type=int, default=2048)
    p.add_argument("--n-ubatch", type=int, default=1024)
    p.add_argument("--n-threads-batch", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--health-window", type=int, default=20)
    p.add_argument("--health-min-json", type=float, default=0.95)
    p.add_argument("--health-min-valid", type=float, default=0.90)
    p.add_argument("--health-min-samples", type=int, default=20)
    p.add_argument("--health-action", default="quarantine_cell")
    p.add_argument("--baseline-profile", default="default_nucleus")
    p.add_argument("--embedding-backend", default="hash")
    p.add_argument("--require-semantic", default="false")
    p.add_argument("--model-path-map", default="")
    p.add_argument("--download-models", action="store_true")
    p.add_argument("--model-dir", default="/content/models/gguf")
    p.add_argument("--hf-model", action="append", default=[], help="model_id=repo_id:filename override/addition")
    p.add_argument("--install-deps", action="store_true", help="Install repo/data deps and llama-cpp-python.")
    p.add_argument("--cuda-llama-cpp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-score", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--append-runs", action="store_true")
    p.add_argument("--append-scores", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.root).resolve()
    output_root = Path(args.output_root)
    runs_dir = output_root / "runs"
    scores_dir = output_root / "scores"
    analysis_dir = output_root / "analysis"
    model_ids = [item.strip() for item in args.models.split(",") if item.strip()]

    if args.install_deps:
        maybe_install_deps(root, cuda=bool(args.cuda_llama_cpp))

    model_path_map = Path(args.model_path_map) if args.model_path_map else output_root / "model_paths.colab.json"
    if args.download_models:
        hf_models = parse_hf_model(args.hf_model)
        paths = download_model_paths(model_ids, Path(args.model_dir), hf_models)
        write_json(model_path_map, paths)
    elif not model_path_map.exists():
        raise SystemExit(
            f"Missing model path map: {model_path_map}. "
            "Pass --download-models or --model-path-map /path/to/model_paths.json."
        )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CREATIVEAI_EMBEDDING_BACKEND"] = args.embedding_backend
    env["CREATIVEAI_REQUIRE_SEMANTIC"] = args.require_semantic
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if env.get("HF_TOKEN") and not env.get("HUGGINGFACE_HUB_TOKEN"):
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]

    session_id = f"phase4-colab-a100-{int(time.time())}"
    cli = [sys.executable, "-m", "creativeai.cli"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        generate_cmd = [
            *cli,
            "generate-grid",
            "--backend",
            args.backend,
            "--model-path-map",
            str(model_path_map),
            "--tasks",
            args.tasks,
            "--methods",
            args.methods,
            "--models",
            args.models,
            "--temperatures",
            args.temperatures,
            "--sampler-profiles",
            args.sampler_profiles,
            "--seeds",
            args.seeds,
            "--top-p",
            "0.9",
            "--max-tokens",
            str(args.max_tokens),
            "--token-budget-per-prompt",
            str(args.token_budget_per_prompt),
            "--quantization",
            "q4_k_m",
            "--strict-json",
            "--max-retries",
            "4",
            "--prompt-mode",
            "auto",
            "--grammar-mode",
            "auto",
            "--compute-tag",
            "phase4_colab_a100",
            "--stage",
            "phase4",
            "--cue-offset",
            str(args.cue_offset),
            "--aut-offset",
            str(args.aut_offset),
            "--limit-cues",
            str(args.limit_cues),
            "--limit-aut",
            str(args.limit_aut),
            "--n-gpu-layers",
            str(args.n_gpu_layers),
            "--n-ctx",
            str(args.n_ctx),
            "--n-threads",
            str(args.n_threads),
            "--n-batch",
            str(args.n_batch),
            "--n-ubatch",
            str(args.n_ubatch),
            "--n-threads-batch",
            str(args.n_threads_batch),
            "--progress",
            "--progress-every",
            str(args.progress_every),
            "--health-window",
            str(args.health_window),
            "--health-min-json",
            str(args.health_min_json),
            "--health-min-valid",
            str(args.health_min_valid),
            "--health-min-samples",
            str(args.health_min_samples),
            "--health-action",
            args.health_action,
            "--health-events",
            str(runs_dir / "health_events.jsonl"),
            "--session-id",
            session_id,
            "--output-dir",
            str(runs_dir),
            "--append-runs" if args.append_runs else "--no-append-runs",
        ]
        run(generate_cmd, env=env, cwd=root)

    if not args.skip_score:
        score_cmd = [
            *cli,
            "score",
            "--input",
            str(runs_dir / "runs.jsonl"),
            "--output-dir",
            str(scores_dir),
            "--require-single-session",
            "--append-scores" if args.append_scores else "--no-append-scores",
        ]
        run(score_cmd, env=env, cwd=root)

    if not args.skip_analysis:
        run(
            [
                *cli,
                "analyze-frontier",
                "--runs",
                str(scores_dir / "scores.jsonl"),
                "--require-single-session",
                "--exclude-invalid",
                "--paired-by",
                "prompt",
                "--compute-matched-by",
                "prompt",
                "--compute-matched-k",
                "1",
                "--compute-matched-token-tolerance",
                "0.20",
                "--output-dir",
                str(analysis_dir),
            ],
            env=env,
            cwd=root,
        )
        run(
            [
                *cli,
                "analyze-samplers",
                "--scores",
                str(scores_dir / "scores.jsonl"),
                "--baseline-profile",
                args.baseline_profile,
                "--require-single-session",
                "--output-dir",
                str(analysis_dir),
            ],
            env=env,
            cwd=root,
        )
        run([*cli, "audit-homogeneity", "--runs", str(runs_dir / "runs.jsonl"), "--output-dir", str(analysis_dir)], env=env, cwd=root)
        run(
            [
                *cli,
                "audit-homogeneity",
                "--runs",
                str(runs_dir / "runs.jsonl"),
                "--by-task",
                "--output-dir",
                str(analysis_dir / "task_stratified"),
            ],
            env=env,
            cwd=root,
        )

    summary = {
        "output_root": str(output_root),
        "runs": str(runs_dir),
        "scores": str(scores_dir),
        "analysis": str(analysis_dir),
        "sampler_report": str(analysis_dir / "SAMPLER_ANALYSIS_REPORT.md"),
        "sampler_json": str(analysis_dir / "sampler_analysis.json"),
        "experiment_manifest": str(runs_dir / "experiment_manifest.json"),
        "model_path_map": str(model_path_map),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
