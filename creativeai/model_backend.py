from __future__ import annotations

import json
import os
import random
import re
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from creativeai.io_utils import file_sha256, stable_hash
from creativeai.schemas import GenerationConfig


class ModelAdapter(Protocol):
    model_id: str
    backend: str

    @property
    def model_hash(self) -> str: ...

    def generate(self, prompt: str, config: GenerationConfig) -> str: ...


_GRAMMAR_CACHE: dict[str, object] = {}


_WORD_BANK = [
    "anchor",
    "beacon",
    "cipher",
    "dynamo",
    "ember",
    "fjord",
    "glyph",
    "harbor",
    "isotope",
    "jungle",
    "keystone",
    "lantern",
    "monsoon",
    "nebula",
    "oasis",
    "paradox",
    "quartz",
    "runway",
    "spectrum",
    "tundra",
    "uplink",
    "vortex",
    "wildfire",
    "xylem",
    "yarrow",
    "zephyr",
    "alloy",
    "basil",
    "cobalt",
    "delta",
    "enzyme",
    "falcon",
    "granite",
    "helium",
    "igloo",
    "jasmine",
    "kernel",
    "labyrinth",
    "matrix",
    "nylon",
    "onyx",
    "plasma",
    "quiver",
    "ripple",
    "semaphore",
    "tangent",
    "uranium",
    "vector",
    "windmill",
    "xenon",
    "yacht",
    "zircon",
]

_AUT_TEMPLATES = [
    "Use the {object} as a {function} {context}",
    "Turn the {object} into a {function} {context}",
    "Repurpose the {object} for {function} {context}",
    "Combine the {object} with nearby items to make a {function} {context}",
]

_FUNCTION_BANK = [
    "portable signal marker",
    "temporary filter",
    "stability brace",
    "heat reflector",
    "water saver",
    "teaching prop",
    "sorting tool",
    "measurement aid",
    "emergency latch",
    "light diffuser",
    "noise dampener",
    "direction guide",
    "safety spacer",
    "manual alarm",
]



def _clean_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z]", "", text.lower())
    return token or "cue"



def _extract_quoted_value(prompt: str, key: str) -> str | None:
    pattern = re.compile(rf"{re.escape(key)}:\s*\"([^\"]+)\"", flags=re.IGNORECASE)
    match = pattern.search(prompt)
    if not match:
        return None
    return match.group(1).strip()


@dataclass
class MockModelAdapter:
    model_id: str = "mock-creativeai"
    backend: str = "mock"

    @property
    def model_hash(self) -> str:
        return "mock-model"

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        seed = config.seed + stable_hash(self.model_id + prompt) % (2**31)
        rng = random.Random(seed)

        lower_prompt = prompt.lower()
        if "alternative uses" in lower_prompt or "object:" in lower_prompt:
            obj = _extract_quoted_value(prompt, "Object") or "object"
            context = _extract_quoted_value(prompt, "Context") or "in context"
            ideas: list[str] = []
            function_pool = _FUNCTION_BANK.copy()
            rng.shuffle(function_pool)
            for idx in range(10):
                template = _AUT_TEMPLATES[idx % len(_AUT_TEMPLATES)]
                function = function_pool[idx % len(function_pool)]
                ideas.append(template.format(object=obj, function=function, context=context))
            return json.dumps(ideas)

        cue = _extract_quoted_value(prompt, "Cue")
        if cue:
            base = _clean_token(cue)
            suffixes = [
                "signal",
                "matrix",
                "harbor",
                "vector",
                "kernel",
                "orbit",
                "network",
                "archive",
                "channel",
                "lattice",
            ]
            rng.shuffle(suffixes)
            words = [f"{base}{suffixes[i]}" for i in range(10)]
            return json.dumps(words)

        bank = _WORD_BANK.copy()
        rng.shuffle(bank)
        return json.dumps(bank[:10])


@dataclass
class LlamaCppAdapter:
    model_id: str
    model_path: str
    backend: str = "llama_cpp"
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_threads: int = 0
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads_batch: int = 0

    def __post_init__(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "llama-cpp-python is required for backend=llama_cpp. Install with: pip install 'creativeai[llama]'"
            ) from exc

        init_params = set(inspect.signature(Llama.__init__).parameters)
        resolved_threads = self.n_threads if self.n_threads > 0 else _default_thread_count()
        resolved_threads_batch = self.n_threads_batch if self.n_threads_batch > 0 else resolved_threads

        kwargs: dict[str, object] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "seed": 0,
            "verbose": False,
            "n_threads": resolved_threads,
        }
        if self.n_batch > 0:
            kwargs["n_batch"] = self.n_batch
        if self.n_ubatch > 0 and "n_ubatch" in init_params:
            kwargs["n_ubatch"] = self.n_ubatch
        if resolved_threads_batch > 0 and "n_threads_batch" in init_params:
            kwargs["n_threads_batch"] = resolved_threads_batch

        self._llama = Llama(**kwargs)

    @property
    def model_hash(self) -> str:
        path = Path(self.model_path)
        if not path.exists():
            return "missing-model"
        return file_sha256(path)[:16]

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        gen_kwargs: dict[str, object] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
        }
        if config.stop:
            gen_kwargs["stop"] = config.stop

        grammar = None
        if config.strict_json:
            grammar_text = _select_json_grammar(config.grammar_mode, count=10)
            try:
                grammar = _get_cached_grammar(grammar_text)
            except Exception as exc:
                raise RuntimeError(f"Failed to compile strict JSON grammar: {exc}") from exc

        prompt_mode = _resolve_prompt_mode(self.model_id, config.prompt_mode)

        # For strict JSON, keep completion path so grammar-constrained decoding is always applied.
        use_chat = prompt_mode == "chat" and not config.strict_json
        if use_chat:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "Return only a JSON array of strings and no extra text.",
                    },
                    {"role": "user", "content": prompt},
                ]
                chat_kwargs = dict(gen_kwargs)
                if grammar is not None:
                    chat_kwargs["grammar"] = grammar
                result = self._llama.create_chat_completion(messages=messages, **chat_kwargs)
                return str(result["choices"][0]["message"]["content"])
            except Exception:
                pass

        completion_kwargs = dict(gen_kwargs)
        if grammar is not None:
            completion_kwargs["grammar"] = grammar
        try:
            result = self._llama.create_completion(prompt=prompt, **completion_kwargs)
        except TypeError as exc:
            if config.strict_json:
                raise RuntimeError("llama_cpp create_completion does not support grammar in this build") from exc
            completion_kwargs.pop("grammar", None)
            result = self._llama.create_completion(prompt=prompt, **completion_kwargs)
        return str(result["choices"][0]["text"])


def _json_array_grammar_exact_count(count: int = 10) -> str:
    if count <= 0:
        return """
root ::= "[" "]"
""".strip()
    sequence = " ".join(["string"] + ['"," string'] * (count - 1))
    return """
root ::= "[" elements "]"
elements ::= __SEQUENCE__
string ::= "\\"" c1 c2 c3 tail "\\""
c1 ::= [A-Za-z0-9]
c2 ::= [A-Za-z0-9-]
c3 ::= [A-Za-z0-9-]
tail ::= [A-Za-z0-9 .,;:!?()'/-]*
""".replace("__SEQUENCE__", sequence).strip()


def _json_word_array_grammar_exact_count(count: int = 10) -> str:
    if count <= 0:
        return 'root ::= "[" "]"'
    sequence = " ".join(["qword"] + ['"," qword'] * (count - 1))
    return """
root ::= "[" elements "]"
elements ::= __SEQUENCE__
qword ::= "\\"" word "\\""
word ::= [A-Za-z] [A-Za-z-]*
""".replace("__SEQUENCE__", sequence).strip()


def _json_idea_array_grammar_exact_count(count: int = 10) -> str:
    if count <= 0:
        return 'root ::= "[" "]"'
    sequence = " ".join(["qidea"] + ['"," qidea'] * (count - 1))
    return """
root ::= "[" elements "]"
elements ::= __SEQUENCE__
qidea ::= "\\"" i1 i2 i3 i4 tail "\\""
i1 ::= [A-Za-z0-9]
i2 ::= [A-Za-z0-9 .,;:!?()'/-]
i3 ::= [A-Za-z0-9 .,;:!?()'/-]
i4 ::= [A-Za-z0-9 .,;:!?()'/-]
tail ::= [A-Za-z0-9 .,;:!?()'/-]*
""".replace("__SEQUENCE__", sequence).strip()


def _select_json_grammar(grammar_mode: str, count: int = 10) -> str:
    mode = (grammar_mode or "auto").strip().lower()
    if mode == "word_list":
        return _json_word_array_grammar_exact_count(count)
    if mode == "idea_list":
        return _json_idea_array_grammar_exact_count(count)
    if mode == "auto":
        return _json_array_grammar_exact_count(count)
    raise ValueError(f"Unsupported grammar_mode: {grammar_mode}")


def _get_cached_grammar(grammar_text: str) -> object:
    from llama_cpp import LlamaGrammar  # type: ignore

    cached = _GRAMMAR_CACHE.get(grammar_text)
    if cached is not None:
        return cached
    compiled = LlamaGrammar.from_string(grammar_text)
    _GRAMMAR_CACHE[grammar_text] = compiled
    return compiled


def _is_instruct_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return ("instruct" in lowered) or lowered.endswith("-it") or lowered.endswith("_it")


def _resolve_prompt_mode(model_id: str, prompt_mode: str) -> str:
    mode = (prompt_mode or "auto").strip().lower()
    if mode == "auto":
        return "chat" if _is_instruct_model(model_id) else "completion"
    if mode in {"chat", "completion"}:
        return mode
    return "completion"


def _default_thread_count(reserve_cores: int = 2) -> int:
    cores = os.cpu_count() or 8
    return max(1, cores - reserve_cores)



def create_model_adapter(
    model_id: str,
    backend: str,
    model_path: str | None = None,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_threads: int = 0,
    n_batch: int = 512,
    n_ubatch: int = 512,
    n_threads_batch: int = 0,
) -> ModelAdapter:
    backend_norm = backend.strip().lower().replace(".", "_")
    if backend_norm == "mock":
        return MockModelAdapter(model_id=model_id)
    if backend_norm in {"llama_cpp", "llamacpp"}:
        if not model_path:
            raise ValueError("backend=llama_cpp requires --model-path")
        return LlamaCppAdapter(
            model_id=model_id,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_threads_batch=n_threads_batch,
        )
    raise ValueError(f"Unsupported backend: {backend}")
