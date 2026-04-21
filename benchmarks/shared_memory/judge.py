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


def _normalize_judge_keys(data: dict) -> dict:
    """Normalize LLM judge response keys to the expected 3-score format."""
    key_map = {
        "profile_usage_score": "profile_usage_score",
        "profile_usage": "profile_usage_score",
        "profile usage": "profile_usage_score",
        "task_usage_score": "task_usage_score",
        "task_usage": "task_usage_score",
        "task usage": "task_usage_score",
        "integration_score": "integration_score",
        "integration": "integration_score",
        "profile_usage_reasoning": "profile_usage_reasoning",
        "task_usage_reasoning": "task_usage_reasoning",
        "integration_reasoning": "integration_reasoning",
        "generic_penalty": "generic_penalty",
        "reasoning": None,
    }
    normalized: Dict[str, Any] = {}
    for k, v in data.items():
        canonical = key_map.get(k.lower().strip())
        if canonical and not isinstance(v, (dict, list)):
            normalized[canonical] = v
        elif canonical is None and isinstance(v, dict):
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
            "Profile Usage (1-5):\n"
            "  5 = Correctly and specifically references multiple profile "
            "attributes (tools, language, style) in the recommendation\n"
            "  4 = Correctly references most profile attributes\n"
            "  3 = References some profile attributes but misses key ones\n"
            "  2 = Vague or incorrect references to profile attributes\n"
            "  1 = No evidence of profile knowledge; response could apply "
            "to any developer\n\n"
            "Task Usage (1-5):\n"
            "  5 = Correctly and specifically references project goals, "
            "blockers, and next steps in the recommendation\n"
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
            "--- GENERIC PENALTY ---\n"
            "If the assistant response does not reference ANY specific profile attribute\n"
            "(tool name, language, style) AND does not reference ANY specific task detail\n"
            "(project name, goal, blocker, next step), the response is considered generic.\n"
            "Set generic_penalty to true if the response is generic, false otherwise.\n\n"
            "Return your scores and reasoning as JSON."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator assessing the quality "
                    "of an AI assistant's response."
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
                        "generic_penalty": {"type": "boolean"},
                        "profile_usage_reasoning": {"type": "string"},
                        "task_usage_reasoning": {"type": "string"},
                        "integration_reasoning": {"type": "string"},
                    },
                    "required": [
                        "profile_usage_score",
                        "task_usage_score",
                        "integration_score",
                        "generic_penalty",
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
                generic_penalty=data.get("generic_penalty"),
                profile_usage_reasoning=data.get("profile_usage_reasoning"),
                task_usage_reasoning=data.get("task_usage_reasoning"),
                integration_reasoning=data.get("integration_reasoning"),
            )

            # Apply generic penalty cap
            if scores.generic_penalty is True:
                scores.profile_usage_score = min(scores.profile_usage_score or 2, 2)
                scores.task_usage_score = min(scores.task_usage_score or 2, 2)
                scores.integration_score = min(scores.integration_score or 2, 2)

            return scores

        except Exception as e:
            logger.warning("Judge evaluation failed: %s", e)
            return JudgeScores()
