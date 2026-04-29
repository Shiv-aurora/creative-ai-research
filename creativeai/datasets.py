from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CueSpec:
    cue: str
    category: str


@dataclass(frozen=True)
class AUTPrompt:
    object_name: str
    context: str


_CUE_CATEGORIES: dict[str, list[str]] = {
    "nature": [
        "forest",
        "river",
        "volcano",
        "desert",
        "coral",
        "mountain",
        "rain",
        "thunder",
        "glacier",
        "meadow",
        "tornado",
        "island",
        "canyon",
        "reef",
        "aurora",
    ],
    "technology": [
        "server",
        "sensor",
        "drone",
        "algorithm",
        "robot",
        "battery",
        "quantum",
        "satellite",
        "firewall",
        "circuit",
        "database",
        "compiler",
        "browser",
        "network",
        "encryption",
    ],
    "health": [
        "vaccine",
        "surgery",
        "therapy",
        "nutrition",
        "sleep",
        "rehab",
        "diagnosis",
        "antibody",
        "clinic",
        "epidemic",
        "wellness",
        "genetics",
        "immunity",
        "pain",
        "metabolism",
    ],
    "education": [
        "classroom",
        "homework",
        "curriculum",
        "tutor",
        "seminar",
        "exam",
        "lecture",
        "laboratory",
        "debate",
        "literacy",
        "pedagogy",
        "campus",
        "scholarship",
        "mentor",
        "syllabus",
    ],
    "economy": [
        "inflation",
        "startup",
        "budget",
        "tax",
        "market",
        "trade",
        "currency",
        "logistics",
        "warehouse",
        "retail",
        "contract",
        "invoice",
        "insurance",
        "portfolio",
        "credit",
    ],
    "society": [
        "election",
        "migration",
        "justice",
        "privacy",
        "community",
        "policy",
        "protest",
        "ethics",
        "identity",
        "culture",
        "language",
        "housing",
        "transport",
        "safety",
        "inclusion",
    ],
    "arts": [
        "poetry",
        "sculpture",
        "melody",
        "theater",
        "cinema",
        "rhythm",
        "canvas",
        "choreography",
        "novel",
        "photography",
        "design",
        "improv",
        "harmony",
        "storytelling",
        "calligraphy",
    ],
    "space": [
        "orbit",
        "asteroid",
        "galaxy",
        "telescope",
        "cosmos",
        "nebula",
        "lunar",
        "rocket",
        "gravity",
        "eclipse",
        "solstice",
        "trajectory",
        "exoplanet",
        "astronaut",
        "radiation",
    ],
}

_AUT_OBJECTS = [
    "brick",
    "paperclip",
    "plastic bottle",
    "umbrella",
    "shoe",
    "blanket",
    "rope",
    "bucket",
    "mirror",
    "rubber band",
    "cardboard box",
    "spoon",
    "coin",
    "pencil",
    "newspaper",
    "towel",
    "backpack",
    "ceramic mug",
    "wooden stick",
    "flashlight",
    "tape",
    "old phone",
    "tennis ball",
    "scarf",
    "keyboard",
    "water hose",
    "ladder",
    "helmet",
    "notebook",
    "paint brush",
]

_AUT_CONTEXTS = [
    "during a power outage",
    "in a classroom with limited supplies",
    "during emergency flood response",
]



def default_cdat_cues() -> list[CueSpec]:
    cues: list[CueSpec] = []
    for category, words in _CUE_CATEGORIES.items():
        for cue in words:
            cues.append(CueSpec(cue=cue, category=category))
    return cues



def default_aut_prompts() -> list[AUTPrompt]:
    rows: list[AUTPrompt] = []
    for obj in _AUT_OBJECTS:
        for context in _AUT_CONTEXTS:
            rows.append(AUTPrompt(object_name=obj, context=context))
    return rows
