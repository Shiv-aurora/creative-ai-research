from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt_template: str
    constraints: list[str]
    expected_format: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, variables: dict[str, Any] | None = None, extra_constraints: list[str] | None = None) -> str:
        vars_in = variables or {}
        base = self.prompt_template.format(**vars_in)
        constraints = list(self.constraints)
        if extra_constraints:
            constraints.extend(extra_constraints)
        if not constraints:
            return base
        return f"{base}\n\nConstraints:\n" + "\n".join(f"- {item}" for item in constraints)


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str
    backend: str
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    quantization: str = "q4_k_m"
    strict_json: bool = True
    max_retries: int = 2
    prompt_mode: Literal["completion", "chat", "auto"] = "auto"
    grammar_mode: Literal["auto", "word_list", "idea_list"] = "auto"
    stop: list[str] = field(default_factory=list)
    n_ctx: int = 4096
    n_threads: int = 0
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads_batch: int = 0
    token_budget_per_prompt: int | None = None
    compute_tag: str = ""
    adaptive_stop_delta: float | None = None
    adaptive_min_iters: int = 1
    trigger_objective: float | None = None


@dataclass
class RunManifest:
    run_id: str
    git_hash: str
    model_hash: str
    quantization: str
    backend: str
    created_at_utc: str
    host: str
    session_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    method: str
    model_id: str
    config: dict[str, Any]
    prompt: str
    output: list[str]
    raw_trace: list[dict[str, Any]]
    validity_flags: dict[str, Any]
    timestamp_utc: str
    token_count: int
    manifest: dict[str, Any]
    session_id: str = ""
    parse_mode: str = "unknown"
    json_valid: bool = False
    retry_count: int = 0
    candidate_objectives: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    effective_calls: int = 1
    compute_group_id: str = ""
    phase3_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        return cls(**payload)


@dataclass
class ScoreRecord:
    run_id: str
    task_id: str
    method: str
    model_id: str
    novelty: float
    appropriateness: float
    usefulness: float
    validity_flags: dict[str, Any]
    metrics: dict[str, Any]
    valid_for_primary: bool = False
    metric_backend: str = "hash"
    lexical_overlap_ratio: float = 0.0
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score_per_1k_tokens: float = 0.0
    compute_matched_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreRecord":
        return cls(**payload)


@dataclass
class FrontierPoint:
    model_id: str
    method: str
    task_group: str
    novelty_mean: float
    appropriateness_mean: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
