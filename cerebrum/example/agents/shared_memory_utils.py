"""Shared constants and utilities for the multi-agent personalization system.

This module provides memory metadata field names, sharing policy values,
memory type values, and helper functions used by the Assistant Agent,
Profile Agent, and Task Agent.
"""

from typing import Any, Optional, List, Dict

# --- Memory metadata field names ---
FIELD_OWNER_AGENT = "owner_agent"
FIELD_USER_ID = "user_id"
FIELD_MEMORY_TYPE = "memory_type"
FIELD_SHARING_POLICY = "sharing_policy"

# --- Sharing policy values ---
POLICY_PRIVATE = "private"
POLICY_SHARED = "shared"

# --- Memory type values ---
MEMORY_TYPE_CONVERSATION = "conversation"
MEMORY_TYPE_PROFILE = "profile"
MEMORY_TYPE_TASK_CONTEXT = "task_context"


def build_memory_metadata(
    owner_agent: str,
    user_id: str,
    memory_type: str,
    sharing_policy: str = POLICY_PRIVATE,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a metadata dict conforming to the memory metadata schema.

    Args:
        owner_agent: The agent_name of the creating agent.
        user_id: Identifier for the user this memory pertains to.
        memory_type: One of MEMORY_TYPE_* constants.
        sharing_policy: POLICY_PRIVATE (default) or POLICY_SHARED.
        **extra: Additional provider-specific keys (e.g., mem0 user_id).

    Returns:
        A metadata dictionary ready to pass to create_memory / update_memory.
    """
    metadata: Dict[str, Any] = {
        FIELD_OWNER_AGENT: owner_agent,
        FIELD_USER_ID: user_id,
        FIELD_MEMORY_TYPE: memory_type,
        FIELD_SHARING_POLICY: sharing_policy,
    }
    metadata.update(extra)
    return metadata


def filter_shared_memories(
    search_results: List[Dict[str, Any]],
    memory_type: Optional[str] = None,
    exclude_owner: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter search results to only shared memories, optionally by type.

    Args:
        search_results: Raw list from search_memories response.
        memory_type: If provided, only include memories of this type.
        exclude_owner: If provided, exclude memories owned by this agent.

    Returns:
        Filtered list of memory result dicts.
    """
    filtered: List[Dict[str, Any]] = []
    for mem in search_results:
        meta = mem.get("metadata", {})
        if meta.get(FIELD_SHARING_POLICY) != POLICY_SHARED:
            continue
        if memory_type and meta.get(FIELD_MEMORY_TYPE) != memory_type:
            continue
        if exclude_owner and meta.get(FIELD_OWNER_AGENT) == exclude_owner:
            continue
        filtered.append(mem)
    return filtered
