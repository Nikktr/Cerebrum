"""LLM-as-judge evaluation for the shared memory evaluation harness.

Uses ``llm_chat_with_json_output`` to score assistant responses on
profile usage, task-context usage, and integration using a structured
3-score rubric.
"""

import json
import logging
from typing import Any, Dict, List

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.config.config_manager import config

from benchmarks.shared_memory.models import (
    JudgeScores,
    SyntheticProfile,
    SyntheticTaskContext,
)
from benchmarks.shared_memory.synth import _unwrap_nested

logger = logging.getLogger(__name__)

# Canonical field names for the 3-score rubric
_SCORE_FIELDS = [
    "profile_usage_score",
    "task_usage_score",
    "integration_score",
]


def _canonicalize_key(raw: str) -> str:
    """Convert an arbitrary key string to snake_case without trailing _score."""
    import re
    # Strip whitespace
    key = raw.strip()
    # Insert underscore before capitals (CamelCase -> Camel_Case)
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    # Lowercase everything
    key = key.lower()
    # Replace spaces and hyphens with underscores
    key = re.sub(r"[\s\-]+", "_", key)
    # Collapse multiple underscores
    key = re.sub(r"_+", "_", key)
    # Strip trailing _score suffix for matching
    key = re.sub(r"_score$", "", key)
    return key


# Map canonicalized base names to their canonical output keys
_CANONICAL_MAP: Dict[str, str] = {
    "profile_usage": "profile_usage_score",
    "task_usage": "task_usage_score",
    "integration": "integration_score",
    "profile_usage_reasoning": "profile_usage_reasoning",
    "task_usage_reasoning": "task_usage_reasoning",
    "integration_reasoning": "integration_reasoning",
}


def _normalize_judge_keys(data: dict) -> dict:
    """Normalize LLM judge response keys to the expected 3-score format.

    Handles variant key formats: snake_case, CamelCase, Title Case,
    hyphenated, UPPER_SNAKE, with or without ``_score`` suffix.
    """
    normalized: Dict[str, Any] = {}
    for k, v in data.items():
        canon = _canonicalize_key(k)
        target = _CANONICAL_MAP.get(canon)
        if target and not isinstance(v, (dict, list)):
            normalized[target] = v
        elif target is None and isinstance(v, dict):
            # Nested reasoning dict — extract by keyword
            for rk, rv in v.items():
                rk_lower = rk.lower().strip()
                if "profile" in rk_lower and "profile_usage_reasoning" not in normalized:
                    normalized["profile_usage_reasoning"] = str(rv)
                elif "task" in rk_lower and "task_usage_reasoning" not in normalized:
                    normalized["task_usage_reasoning"] = str(rv)
                elif "integrat" in rk_lower and "integration_reasoning" not in normalized:
                    normalized["integration_reasoning"] = str(rv)
    return normalized


def _clamp_score(value: Any, name: str) -> int:
    """Clamp a score to [1, 5], logging a warning if out of range."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        logger.warning("%s is not an integer (%s), defaulting to 1", name, value)
        return 1
    if v < 1 or v > 5:
        logger.warning("%s %d out of range, clamping to [1, 5]", name, v)
    return max(1, min(5, v))


class LLMJudge:
    """Evaluates assistant responses using a 3-score rubric."""

    def __init__(self, agent_name: str = "eval_judge"):
        self.agent_name = agent_name
        self.kernel_url = config.get_kernel_url()

    def _build_judge_prompt(
        self,
        query: str,
        response: str,
        profile: SyntheticProfile,
        task_context: SyntheticTaskContext,
        plausible_actions: list[str] | None = None,
    ) -> List[Dict[str, str]]:
        """Build the messages list for the LLM judge call."""
        # Build plausible actions section (only when provided and non-empty)
        plausible_actions_section = ""
        if plausible_actions:
            actions_list = "\n".join(
                f"  {i}. {action}" for i, action in enumerate(plausible_actions, 1)
            )
            plausible_actions_section = (
                "--- PLAUSIBLE ACTIONS ---\n"
                "The developer had these pending options to choose from:\n"
                f"{actions_list}\n\n"
            )

        user_content = (
            "Evaluate the following AI assistant response.\n\n"
            "--- USER PROFILE ---\n"
            f"Name: {profile.user_name}\n"
            f"Preferred Tools: {', '.join(profile.preferred_tools)}\n"
            f"Preferred Language: {profile.preferred_language}\n"
            f"Response Style: {profile.response_style}\n\n"
            "--- TASK CONTEXT ---\n"
            f"Current Project: {task_context.current_project}\n"
            f"Active Experiment: {task_context.active_experiment}\n"
            f"Goals: {', '.join(task_context.goals)}\n"
            f"Blockers: {', '.join(task_context.blockers)}\n"
            f"Next Steps: {', '.join(task_context.next_steps)}\n\n"
            f"{plausible_actions_section}"
            "--- FOLLOW-UP QUERY ---\n"
            f"{query}\n\n"
            "--- ASSISTANT RESPONSE ---\n"
            f"{response}\n\n"
            "--- SCORING RUBRIC ---\n"
            "Score based on whether the response content references "
            "profile and task attributes, regardless of response length. "
            "A short response that correctly references the user's "
            "preferred tools and current project is well-personalized.\n\n"
            "Profile Usage (1-5):\n"
            "  5 = Correctly and specifically references multiple profile "
            "attributes (tools, language, style) in the recommendation, "
            "regardless of response length\n"
            "  4 = Correctly references most profile attributes\n"
            "  3 = References some profile attributes but misses key ones\n"
            "  2 = Vague or incorrect references to profile attributes\n"
            "  1 = No evidence of profile knowledge; response could apply "
            "to any developer\n\n"
            "Task Usage (1-5):\n"
            "  5 = Correctly and specifically references project goals, "
            "blockers, and next steps in the recommendation, "
            "regardless of response length\n"
            "  4 = Correctly references most task context details\n"
            "  3 = References some task context details but misses key ones\n"
            "  2 = Vague or incorrect references to task context\n"
            "  1 = No evidence of task context knowledge; response is "
            "generic advice\n\n"
            "Integration (1-5):\n"
            "  5 = Seamlessly combines profile preferences and task context "
            "into a single grounded recommendation\n"
            "  4 = Combines both sources with minor gaps in integration\n"
            "  3 = Addresses profile and task context separately without "
            "integrating them\n"
            "  2 = Mentions both sources but the recommendation does not "
            "logically follow from them\n"
            "  1 = No integration; response addresses at most one source "
            "or is entirely generic\n\n"
            "Return your scores and reasoning as JSON."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator assessing the quality "
                    "of an AI assistant's response. "
                    "The assistant had access to shared memories injected "
                    "by the kernel containing the user's profile and task "
                    "context. A concise response that demonstrates awareness "
                    "of these attributes is well-personalized, not generic."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def evaluate(
        self,
        query: str,
        response: str,
        profile: SyntheticProfile,
        task_context: SyntheticTaskContext,
        plausible_actions: list[str] | None = None,
    ) -> JudgeScores:
        """Score an assistant response on profile usage, task usage, and integration."""
        messages = self._build_judge_prompt(
            query, response, profile, task_context, plausible_actions
        )

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_scores",
                "schema": {
                    "type": "object",
                    "properties": {
                        "profile_usage_score": {"type": "integer"},
                        "task_usage_score": {"type": "integer"},
                        "integration_score": {"type": "integer"},
                        "profile_usage_reasoning": {"type": "string"},
                        "task_usage_reasoning": {"type": "string"},
                        "integration_reasoning": {"type": "string"},
                    },
                    "required": [
                        "profile_usage_score",
                        "task_usage_score",
                        "integration_score",
                        "profile_usage_reasoning",
                        "task_usage_reasoning",
                        "integration_reasoning",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        try:
            llm_response = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                base_url=self.kernel_url,
                response_format=response_format,
            )

            raw = llm_response["response"]["response_message"]
            data = json.loads(raw) if isinstance(raw, str) else raw
            data = _normalize_judge_keys(data)

            pu = data.get("profile_usage_score")
            tu = data.get("task_usage_score")
            ig = data.get("integration_score")

            if pu is None or tu is None or ig is None:
                logger.warning("Judge returned incomplete scores: %s", data)
                return JudgeScores()

            scores = JudgeScores(
                profile_usage_score=_clamp_score(pu, "profile_usage_score"),
                task_usage_score=_clamp_score(tu, "task_usage_score"),
                integration_score=_clamp_score(ig, "integration_score"),
                profile_usage_reasoning=data.get("profile_usage_reasoning"),
                task_usage_reasoning=data.get("task_usage_reasoning"),
                integration_reasoning=data.get("integration_reasoning"),
            )

            return scores

        except Exception as e:
            logger.warning("Judge evaluation failed: %s", e)
            return JudgeScores()
