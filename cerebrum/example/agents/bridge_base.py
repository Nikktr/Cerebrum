import json
import sys
import os
import requests as _requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from cerebrum.memory.apis import create_memory, search_memories
from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    filter_shared_memories,
    POLICY_SHARED,
    MEMORY_TYPE_TASK_CONTEXT,
)

KERNEL_URL = "http://localhost:8000"
DEFAULT_PROJECT = "global"


class BridgeAgent:
    """Base class for CLI bridge agents with project-scoped shared memory."""

    def __init__(self, agent_name):
        self.agent_name = agent_name

    def _resolve_project(self, task_input):
        """Extract project_id, project_path, task from task_input."""
        if isinstance(task_input, dict):
            project_id = task_input.get("project_id", DEFAULT_PROJECT)
            project_path = task_input.get("project_path", "")
            task = task_input.get("task", str(task_input))
            return project_id, project_path, task
        return DEFAULT_PROJECT, "", str(task_input)

    def _get_shared_context(self, task_text, project_id):
        try:
            response = search_memories(
                agent_name=self.agent_name,
                query=task_text,
                k=5,
                base_url=KERNEL_URL,
                user_id=project_id,
                sharing_policy="shared",
            )
            if not response or not hasattr(response, "search_results"):
                return ""
            results = response.search_results or []
            relevant = filter_shared_memories(
                results, exclude_owner=self.agent_name
            )
            if not relevant:
                return ""
            lines = []
            for mem in relevant[:3]:
                owner = mem.get("metadata", {}).get("owner_agent", "unknown")
                content = mem.get("content", "")
                if content:
                    lines.append(f"[{owner}]: {content[:500]}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _save_to_memory(self, task_text, result_text, project_id):
        try:
            content = f"Task: {task_text[:200]}\nResult: {result_text[:1000]}"
            metadata = build_memory_metadata(
                owner_agent=self.agent_name,
                user_id=project_id,
                memory_type=MEMORY_TYPE_TASK_CONTEXT,
                sharing_policy=POLICY_SHARED,
            )
            create_memory(
                agent_name=self.agent_name,
                content=content,
                metadata=metadata,
                base_url=KERNEL_URL,
            )
        except Exception:
            pass

    def _get_mcp_tools(self, project_id):
        """Get list of MCP tools available for this project."""
        try:
            r = _requests.get(
                f"{KERNEL_URL}/mcp/project/{project_id}/tools", timeout=10
            )
            data = r.json()
            if data.get("status") != "success":
                return {}
            return data.get("servers", {})
        except Exception:
            return {}

    def _call_mcp_tool(self, server, tool, arguments=None):
        """Call an MCP tool through the kernel proxy."""
        try:
            payload = {"server": server, "tool": tool}
            if arguments is not None:
                payload["arguments"] = arguments
            r = _requests.post(
                f"{KERNEL_URL}/mcp/call", json=payload, timeout=30
            )
            data = r.json()
            if data.get("status") == "success":
                result = data.get("result", {})
                content = result.get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else json.dumps(result)
            return f"MCP error: {data}"
        except Exception as e:
            return f"MCP call failed: {e}"

    def _format_mcp_tools_for_prompt(self, mcp_tools):
        """Format MCP tools into a text block for the agent prompt."""
        if not mcp_tools:
            return ""
        lines = ["Available MCP tools (call via AIOS kernel):"]
        for server, tools in mcp_tools.items():
            if not tools or (len(tools) == 1 and "error" in tools[0]):
                continue
            lines.append(f"  Server: {server}")
            for t in tools:
                name = t.get("name", "?")
                desc = t.get("description", "")
                lines.append(f"    - {name}: {desc[:80]}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_prompt_with_context(self, task_text, context, mcp_tools=None, project_path=""):
        parts = [task_text]
        if context:
            parts.append(f"Context from other agents:\n{context}")
        if mcp_tools:
            tool_text = self._format_mcp_tools_for_prompt(mcp_tools)
            if tool_text:
                parts.append(tool_text)
        if project_path:
            parts.append(f"[System note: working directory is {project_path}]")
        return "\n\n".join(parts)
