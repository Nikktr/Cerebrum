"""Agent execution pipeline for the shared memory evaluation harness.

Handles loading agents locally and running them in sequence:
ProfileAgent → TaskAgent → AssistantAgent. The share_memory flag
controls whether agents store memories as shared or private,
corresponding to the Phase 2 and Phase 1 experimental conditions.

The pipeline captures injection diagnostics from the kernel response
(when auto_inject is enabled) and tracks written memory metadata
by intercepting create_memory calls from the harness side.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

logger = logging.getLogger(__name__)

from cerebrum.example.agents.profile_agent.agent import ProfileAgent
from cerebrum.example.agents.task_agent.agent import TaskAgent
from cerebrum.example.agents.assistant_agent.agent import AssistantAgent
from cerebrum.memory.apis import search_memories

from benchmarks.shared_memory.models import (
    InjectedMemoryEntry,
    InjectionDiagnostics,
    RetrievalLog,
    RetrievalLogEntry,
    SyntheticTrialData,
    WrittenMemoryRecord,
)


@dataclass
class PipelineResult:
    """Result from a single trial's agent pipeline execution."""

    profile_result: dict
    task_result: dict
    assistant_result: dict
    assistant_response: str
    latency_seconds: float
    retrieval_log: Optional[RetrievalLog] = None
    injection_diagnostics: Optional[InjectionDiagnostics] = None
    written_memories: List[WrittenMemoryRecord] = field(default_factory=list)


class AgentPipeline:
    """Runs the three-agent pipeline for a single trial.

    Instantiates ProfileAgent, TaskAgent, and AssistantAgent in sequence,
    configuring the share_memory attribute based on the experimental
    condition. Measures AssistantAgent latency for metric collection.

    Instead of patching search_memories on the agent side, the pipeline:
    - Patches create_memory to capture WrittenMemoryRecord entries
    - Extracts injection diagnostics from the kernel's llm_chat response
    - Falls back to a harness-side search_memories audit query when the
      kernel does not return diagnostics

    Args:
        share_memory: If True, agents use sharing_policy="shared" (Phase 2).
            If False, agents use sharing_policy="private" (Phase 1).
    """

    def __init__(self, share_memory: bool):
        self.share_memory = share_memory

    def run_trial(self, trial_data: SyntheticTrialData) -> PipelineResult:
        """Execute the full agent pipeline for one trial.

        Args:
            trial_data: The synthetic data for this trial, containing
                profile, task_context, and follow_up_query.

        Returns:
            PipelineResult with all agent outputs, assistant latency,
            injection diagnostics, and written memory records.
        """
        written_records: List[WrittenMemoryRecord] = []

        # Capture reference to real create_memory before patching
        from cerebrum.memory.apis import create_memory as _real_create_memory

        def capturing_create_memory(agent_name, content, metadata=None, base_url=None):
            """Intercept create_memory to capture written metadata."""
            if metadata:
                written_records.append(WrittenMemoryRecord(
                    agent_name=metadata.get("owner_agent", agent_name),
                    memory_type=metadata.get("memory_type", ""),
                    sharing_policy=metadata.get("sharing_policy", "private"),
                    user_id=metadata.get("user_id", ""),
                ))
            # Call through to the real create_memory (captured before patch)
            return _real_create_memory(agent_name, content, metadata=metadata, base_url=base_url)

        with patch("cerebrum.memory.apis.create_memory", side_effect=capturing_create_memory):
            # Step 1: Run ProfileAgent with synthetic profile data
            profile_agent = ProfileAgent("profile_agent")
            profile_agent.share_memory = self.share_memory
            profile_agent.user_id = trial_data.user_id
            profile_result = profile_agent.run(
                json.dumps(trial_data.profile.model_dump())
            )

            # Step 2: Run TaskAgent with synthetic task context data
            task_agent = TaskAgent("task_agent")
            task_agent.share_memory = self.share_memory
            task_agent.user_id = trial_data.user_id
            task_result = task_agent.run(
                json.dumps(trial_data.task_context.model_dump())
            )

            # Step 3: Run AssistantAgent with follow-up query, measuring latency
            assistant_agent = AssistantAgent("assistant_agent")
            assistant_agent.share_memory = self.share_memory
            assistant_agent.user_id = trial_data.user_id

            start_time = time.time()
            assistant_result = assistant_agent.run(trial_data.follow_up_query)
            latency_seconds = time.time() - start_time

        # Extract the response text from the assistant result
        assistant_response = assistant_result.get("result", "")

        # Try to extract injection diagnostics from the kernel response
        injection_diagnostics = self._extract_injection_diagnostics(assistant_result)

        # Build retrieval log: use kernel diagnostics or fall back to audit query
        if injection_diagnostics and injection_diagnostics.injected_count > 0:
            retrieval_log = self._retrieval_log_from_diagnostics(injection_diagnostics)
            # injection_status defaults to "confirmed"
        else:
            retrieval_log = self._audit_shared_memories(trial_data.user_id)
            if retrieval_log.shared_memory_count > 0:
                retrieval_log.injection_status = "audit_inferred"
            elif self.share_memory:
                retrieval_log.injection_status = "unknown"
                logger.warning(
                    "Observability gap: kernel diagnostics absent and audit "
                    "query returned 0 results for Phase 2 trial. "
                    "Injection status unknown."
                )

        return PipelineResult(
            profile_result=profile_result,
            task_result=task_result,
            assistant_result=assistant_result,
            assistant_response=assistant_response,
            latency_seconds=latency_seconds,
            retrieval_log=retrieval_log,
            injection_diagnostics=injection_diagnostics,
            written_memories=written_records,
        )

    def _extract_injection_diagnostics(
        self, assistant_result: dict
    ) -> Optional[InjectionDiagnostics]:
        """Extract injection diagnostics from the kernel's llm_chat response.

        The kernel may include an ``injection_diagnostics`` field in the
        response when ``auto_inject`` is enabled.

        Args:
            assistant_result: The raw result dict from AssistantAgent.run().

        Returns:
            InjectionDiagnostics if the kernel provided them, else None.
        """
        diag_data = assistant_result.get("injection_diagnostics")
        if not isinstance(diag_data, dict):
            return None

        entries = []
        for mem in diag_data.get("injected_memories", []):
            entries.append(InjectedMemoryEntry(
                owner_agent=mem.get("owner_agent", ""),
                memory_type=mem.get("memory_type", ""),
                match_score=mem.get("match_score"),
            ))

        return InjectionDiagnostics(
            injected_count=diag_data.get("injected_count", len(entries)),
            injected_memories=entries,
        )

    def _retrieval_log_from_diagnostics(
        self, diagnostics: InjectionDiagnostics
    ) -> RetrievalLog:
        """Build a RetrievalLog from kernel injection diagnostics.

        Args:
            diagnostics: The InjectionDiagnostics extracted from the response.

        Returns:
            RetrievalLog populated from the diagnostics data.
        """
        entries = []
        cross_agent = False
        for mem in diagnostics.injected_memories:
            entries.append(RetrievalLogEntry(
                owner_agent=mem.owner_agent,
                memory_type=mem.memory_type,
            ))
            if mem.owner_agent != "assistant_agent":
                cross_agent = True

        return RetrievalLog(
            shared_memory_count=diagnostics.injected_count,
            retrieved_memories=entries,
            cross_agent_found=cross_agent,
        )

    def _audit_shared_memories(self, user_id: str) -> RetrievalLog:
        """Query search_memories from the harness side to audit shared memories.

        When the kernel does not return injection diagnostics (e.g.,
        auto_inject is off or the kernel version doesn't support it),
        the harness performs its own audit query to check what shared
        memories exist for the user.

        Args:
            user_id: The user identifier to scope the audit query.

        Returns:
            RetrievalLog built from the audit query results.
        """
        if not user_id:
            return RetrievalLog()

        try:
            result = search_memories(
                agent_name="assistant_agent",
                query="user context",
                k=20,
                user_id=user_id,
                sharing_policy="shared",
            )
        except Exception:
            return RetrievalLog()

        return self._build_retrieval_log_from_search(result)

    def _build_retrieval_log_from_search(
        self, search_result: dict
    ) -> RetrievalLog:
        """Build a RetrievalLog from a raw search_memories response.

        Args:
            search_result: Raw result dict from search_memories.

        Returns:
            RetrievalLog with shared_memory_count, retrieved_memories,
            and cross_agent_found populated from the search results.
        """
        entries = []
        shared_count = 0
        cross_agent = False

        if not isinstance(search_result, dict):
            return RetrievalLog()

        resp = search_result.get("response", {})
        if not isinstance(resp, dict):
            return RetrievalLog()

        search_results = resp.get("search_results", []) or []
        for mem in search_results:
            meta = mem.get("metadata", {})
            if not meta:
                continue
            owner = meta.get("owner_agent", "")
            mem_type = meta.get("memory_type", "")
            if not owner:
                continue
            entries.append(RetrievalLogEntry(
                owner_agent=owner,
                memory_type=mem_type,
            ))
            if meta.get("sharing_policy") == "shared":
                shared_count += 1
            if owner != "assistant_agent":
                cross_agent = True

        return RetrievalLog(
            shared_memory_count=shared_count,
            retrieved_memories=entries,
            cross_agent_found=cross_agent,
        )
