from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any

from creativeai.schemas import GenerationConfig


SAMPLER_PROFILES: dict[str, dict[str, Any]] = {
    "manual": {},
    "low_temp": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
    },
    "default_nucleus": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
    },
    "high_temp": {
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 0,
        "repeat_penalty": 1.0,
    },
    "spread_topk_minp": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 80,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
    },
    "anti_repetition": {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 60,
        "repeat_penalty": 1.15,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
    },
    "mirostat": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "mirostat_mode": 2,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
    },
    # ── Phase 6 ablation profiles ─────────────────────────────────────────────
    # Each isolates one component of anti_repetition to identify the driver.
    "ablation_temp_only": {
        # Temperature lifted to 0.9; all penalties at default. Tests: does
        # a slightly higher temp alone account for the anti_repetition gain?
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "ablation_repeat_only": {
        # Repeat penalty only at default temp. Tests: does discouraging exact
        # token repetition drive the novelty gain independently of temperature?
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.15,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "ablation_penalties_only": {
        # Frequency + presence penalties at default temp, no repeat penalty.
        # Tests: does penalising frequent/present tokens (diversity pressure)
        # alone shift the frontier?
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
    },
    "anti_repetition_strong": {
        # Pushed further than anti_repetition. If the ablation shows
        # penalties drive the effect, this tests whether more is better
        # or if there's a quality cliff.
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 60,
        "repeat_penalty": 1.25,
        "frequency_penalty": 0.35,
        "presence_penalty": 0.15,
    },
}


DECODING_FINGERPRINT_FIELDS = (
    "sampler_profile",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "frequency_penalty",
    "presence_penalty",
    "mirostat_mode",
    "mirostat_tau",
    "mirostat_eta",
)


def apply_sampler_profile(config: GenerationConfig) -> GenerationConfig:
    profile_name = (config.sampler_profile or "manual").strip()
    if not profile_name:
        profile_name = "manual"
    if profile_name not in SAMPLER_PROFILES:
        known = ", ".join(sorted(SAMPLER_PROFILES))
        raise ValueError(f"Unsupported sampler_profile: {profile_name}. Known profiles: {known}")
    overrides = SAMPLER_PROFILES[profile_name]
    if not overrides:
        return replace(config, sampler_profile=profile_name)
    return replace(config, sampler_profile=profile_name, **overrides)


def sampler_profile_names() -> list[str]:
    return sorted(SAMPLER_PROFILES)


def decoding_settings(config: GenerationConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {field: payload.get(field) for field in DECODING_FINGERPRINT_FIELDS}


def decoding_fingerprint(config: GenerationConfig) -> str:
    settings = decoding_settings(config)
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
