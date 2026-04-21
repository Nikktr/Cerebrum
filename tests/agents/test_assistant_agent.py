"""Unit tests for AssistantAgent refactor — kernel-managed shared memory.

Validates: Requirements 4.1, 4.2, 4.4, 5.1, 5.2, 5.3, 5.4
"""

import sys
sys.path.insert(0, ".")

import inspect
import json
import os
import tempfile
from unittest.mock import patch, MagicMock


def _make_config_dir():
    """Create a temporary config.json for AssistantAgent to load."""
    tmpdir = tempfile.mkdtemp()
    config_data = {
        "name": "test_assistant",
        "description": [
            "You are a personalized assistant agent. ",
            "You help users with their queries.",
        ],
        "tools": [],
        "meta": {"author": "test", "version": "0.0.1", "license": "MIT"},
        "build": {"entry": "agent.py", "module": "AssistantAgent"},
    }
    config_path = os.path.join(tmpdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return tmpdir, config_path


def _create_agent():
    """Instantiate AssistantAgent with a patched config path."""
    from cerebrum.example.agents.assistant_agent.agent import AssistantAgent

    tmpdir, config_path = _make_config_dir()

    with patch.object(AssistantAgent, "load_config") as mock_load:
        with open(config_path, "r") as f:
            mock_load.return_value = json.load(f)
        agent = AssistantAgent(agent_name="test_assistant")
    return agent


def test_system_instruction_has_no_shared_context():
    """run() calls llm_chat with messages that don't contain manually
    retrieved shared memories. (Req 4.1)"""
    agent = _create_agent()

    mock_response = {
        "response": {"response_message": "Hello, how can I help?"}
    }

    with patch(
        "cerebrum.example.agents.assistant_agent.agent.llm_chat",
        return_value=mock_response,
    ) as mock_llm, patch(
        "cerebrum.example.agents.assistant_agent.agent.create_memory",
    ):
        agent.run("What is the weather?")

        # Inspect the messages passed to llm_chat
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
            "messages", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        )

        # System message should only contain the config description,
        # not any "shared memory" / "profile" / "task_context" retrieval text
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"Expected exactly 1 system message, got {len(system_msgs)}"
        )
        system_content = system_msgs[0]["content"]

        # The system instruction should be the config description only
        expected = "You are a personalized assistant agent. You help users with their queries."
        assert system_content == expected, (
            f"System instruction should be config description only.\n"
            f"Got: {system_content!r}"
        )

    print("PASSED: system instruction has no manually retrieved shared context")


def test_search_memories_not_called():
    """search_memories is never called during run(). (Req 4.2)"""
    agent = _create_agent()

    mock_response = {
        "response": {"response_message": "Sure thing."}
    }

    with patch(
        "cerebrum.example.agents.assistant_agent.agent.llm_chat",
        return_value=mock_response,
    ), patch(
        "cerebrum.example.agents.assistant_agent.agent.create_memory",
    ), patch(
        "cerebrum.memory.apis.search_memories",
    ) as mock_search:
        agent.run("Tell me about Python.")

        mock_search.assert_not_called()

    print("PASSED: search_memories is not called during run()")


def test_conversation_memory_stored_with_correct_type():
    """create_memory is called with metadata containing
    memory_type='conversation'. (Req 4.4)"""
    agent = _create_agent()

    mock_response = {
        "response": {"response_message": "Here is the info."}
    }

    with patch(
        "cerebrum.example.agents.assistant_agent.agent.llm_chat",
        return_value=mock_response,
    ), patch(
        "cerebrum.example.agents.assistant_agent.agent.create_memory",
    ) as mock_create:
        agent.run("Summarize my notes.")

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        metadata = (
            call_kwargs.kwargs.get("metadata")
            or call_kwargs[1].get("metadata")
        )
        assert metadata is not None, "create_memory must be called with metadata"
        assert metadata.get("memory_type") == "conversation", (
            f"Expected memory_type='conversation', got {metadata.get('memory_type')!r}"
        )

    print("PASSED: conversation memory stored with memory_type='conversation'")


def test_filter_shared_memories_not_imported():
    """filter_shared_memories is not present in the AssistantAgent module
    namespace. (Req 5.3)"""
    import cerebrum.example.agents.assistant_agent.agent as agent_module

    # Check the module namespace
    assert not hasattr(agent_module, "filter_shared_memories"), (
        "filter_shared_memories should not be imported in assistant_agent module"
    )

    # Also verify via source inspection that there is no import of it
    source = inspect.getsource(agent_module)
    assert "filter_shared_memories" not in source, (
        "filter_shared_memories should not appear in assistant_agent source"
    )

    print("PASSED: filter_shared_memories is not imported in default code path")


def test_filter_shared_memories_importable_and_callable():
    """filter_shared_memories is importable from shared_memory_utils and
    callable with an empty list. (Req 5.1)"""
    from cerebrum.example.agents.shared_memory_utils import filter_shared_memories

    assert callable(filter_shared_memories), (
        "filter_shared_memories should be callable"
    )

    # Call with an empty list — should return an empty list without error
    result = filter_shared_memories([])
    assert result == [], (
        f"filter_shared_memories([]) should return [], got {result!r}"
    )

    print("PASSED: filter_shared_memories is importable and callable")


def test_search_memories_accepts_cross_agent_params():
    """search_memories accepts user_id and sharing_policy keyword
    arguments without raising. (Req 5.2, 5.4)"""
    from cerebrum.memory.apis import search_memories

    mock_response = {
        "response_class": "memory",
        "search_results": [],
        "success": True,
    }

    with patch(
        "cerebrum.memory.apis.send_request",
        return_value=mock_response,
    ) as mock_send:
        # Call with both cross-agent parameters
        search_memories(
            "test_agent",
            "test query",
            k=3,
            user_id="user_42",
            sharing_policy="shared",
        )

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        query_obj = call_args[0][1]  # second positional arg is the query
        params = query_obj.params

        assert params.get("user_id") == "user_42", (
            f"Expected user_id='user_42' in params, got {params.get('user_id')!r}"
        )
        assert params.get("sharing_policy") == "shared", (
            f"Expected sharing_policy='shared' in params, got {params.get('sharing_policy')!r}"
        )

    print("PASSED: search_memories accepts user_id and sharing_policy params")


if __name__ == "__main__":
    test_system_instruction_has_no_shared_context()
    test_search_memories_not_called()
    test_conversation_memory_stored_with_correct_type()
    test_filter_shared_memories_not_imported()
    test_filter_shared_memories_importable_and_callable()
    test_search_memories_accepts_cross_agent_params()
    print("\nAll tests passed.")
