"""Registry of extraction solutions. Add a Solution subclass here to list it on the landing page."""
import importlib
import logging

from app.solutions.base import Solution


logger = logging.getLogger(__name__)

SOLUTION_CLASSES = [
    ("app.solutions.langextract", "LangExtractSolution"),
    ("app.solutions.nuextract", "NuExtractSolution"),
    ("app.solutions.llm", "LLMSolution"),
    ("app.solutions.unstract", "UnstractSolution"),
]


def build_registry() -> dict[str, Solution]:
    registry: dict[str, Solution] = {}
    for module_name, class_name in SOLUTION_CLASSES:
        try:
            solution = getattr(importlib.import_module(module_name), class_name)()
        except Exception as error:  # noqa: BLE001 - one broken solution must not take down the app
            logger.warning("Solution %s unavailable: %s: %s", class_name, type(error).__name__, error)
            continue
        registry[solution.id] = solution
    return registry
