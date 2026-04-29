from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from creativeai.datasets import default_aut_prompts, default_cdat_cues
from creativeai.methods import build_method_runner
from creativeai.model_backend import create_model_adapter
from creativeai.pipeline import generate_run
from creativeai.schemas import GenerationConfig
from creativeai.tasks import build_task

RUNS_DIR = Path("outputs/phase1_real_batch1/runs")
MODEL_MAP_PATH = Path("model_paths.downloaded.json")

CUES = [c.cue for c in default_cdat_cues()[:20]]
AUTS = [(a.object_name, a.context) for a in default_aut_prompts()[:18]]

TARGETS = [
    ("mistral-7b-v0.3", ["restlessness_loop"]),
    ("mistral-7b-instruct-v0.3", ["one_shot", "restlessness_loop"]),
]


def load_existing() -> set[tuple]:
    existing: set[tuple] = set()
    for f in RUNS_DIR.glob("run-*.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue

        model = r.get("model_id")
        method = r.get("method")
        task = r.get("task_id")
        md = r.get("metadata", {}) or {}

        if task == "cdat":
            key = (model, method, task, md.get("cue"))
        elif task == "aut":
            key = (model, method, task, md.get("object"), md.get("context"))
        else:
            key = (model, method, task)
        existing.add(key)
    return existing


def build_queue(existing: set[tuple]) -> list[tuple[str, str, str, object | None]]:
    queue: list[tuple[str, str, str, object | None]] = []
    for model, methods in TARGETS:
        for method in methods:
            if (model, method, "dat") not in existing:
                queue.append((model, method, "dat", None))
            for cue in CUES:
                if (model, method, "cdat", cue) not in existing:
                    queue.append((model, method, "cdat", cue))
            for obj, ctx in AUTS:
                if (model, method, "aut", obj, ctx) not in existing:
                    queue.append((model, method, "aut", (obj, ctx)))
    return queue


def main() -> None:
    if not RUNS_DIR.exists():
        raise SystemExit(f"Missing runs dir: {RUNS_DIR}")
    if not MODEL_MAP_PATH.exists():
        raise SystemExit(f"Missing model map: {MODEL_MAP_PATH}")

    model_map = json.loads(MODEL_MAP_PATH.read_text())
    existing = load_existing()
    queue = build_queue(existing)

    print(f"RESUME_QUEUE {len(queue)}", flush=True)
    if not queue:
        print("Nothing to resume.", flush=True)
        return

    ctr = Counter((m, me, t) for m, me, t, _ in queue)
    for key in sorted(ctr):
        print(f"REMAINING {key[0]} {key[1]} {key[2]} {ctr[key]}", flush=True)

    cfg_by_model = {
        model: GenerationConfig(
            model_id=model,
            backend="llama_cpp",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            quantization="q4_k_m",
        )
        for model, _ in TARGETS
    }

    models = {
        model: create_model_adapter(
            model_id=model,
            backend="llama_cpp",
            model_path=model_map[model],
            n_gpu_layers=-1,
        )
        for model, _ in TARGETS
    }
    methods = {
        "one_shot": build_method_runner("one_shot", restlessness_k=3),
        "restlessness_loop": build_method_runner("restlessness_loop", restlessness_k=3),
    }

    done = 0
    total = len(queue)

    for model, method, task, payload in queue:
        if task == "dat":
            task_spec = build_task("dat")
        elif task == "cdat":
            task_spec = build_task("cdat", cue=str(payload))
        else:
            obj, ctx = payload  # type: ignore[misc]
            task_spec = build_task("aut", obj=obj, context=ctx)

        rec = generate_run(
            task_spec=task_spec,
            method_runner=methods[method],
            model=models[model],
            config=cfg_by_model[model],
            output_dir=RUNS_DIR,
        )
        done += 1
        if done % 3 == 0 or done == total:
            print(
                f"PROGRESS {done}/{total} run_id={rec.run_id} model={model} method={method} task={task}",
                flush=True,
            )

    print(f"RESUME_DONE {done}", flush=True)


if __name__ == "__main__":
    main()
