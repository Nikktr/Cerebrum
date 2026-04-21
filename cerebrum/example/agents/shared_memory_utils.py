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


_VALID_SHARING_POLICIES = {POLICY_PRIVATE, POLICY_SHARED}
_VALID_MEMORY_TYPES = {MEMORY_TYPE_PROFILE, MEMORY_TYPE_TASK_CONTEXT, MEMORY_TYPE_CONVERSATION}


def build_memory_metadata(
    owner_agent: str,
    user_id: str,
    memory_type: str,
    sharing_policy: str = POLICY_PRIVATE,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a validated metadata dict for kernel memory operations.

    Args:
        owner_agent: The agent_name of the creating agent. Must be non-empty.
        user_id: Identifier for the user. Must be non-empty.
        memory_type: One of "profile", "task_context", "conversation".
        sharing_policy: "private" (default) or "shared".
        **extra: Additional provider-specific keys.

    Returns:
        A metadata dictionary with exactly the four standard fields
        plus any extra kwargs.

    Raises:
        ValueError: If sharing_policy, memory_type, owner_agent, or
            user_id is invalid.
    """
    if sharing_policy not in _VALID_SHARING_POLICIES:
        raise ValueError(
            f"Invalid sharing_policy: {sharing_policy!r}. "
            f"Must be one of {sorted(_VALID_SHARING_POLICIES)}."
        )
    if memory_type not in _VALID_MEMORY_TYPES:
        raise ValueError(
            f"Invalid memory_type: {memory_type!r}. "
            f"Must be one of {sorted(_VALID_MEMORY_TYPES)}."
        )
    if not isinstance(owner_agent, str) or not owner_agent:
        raise ValueError(
            f"Invalid owner_agent: {owner_agent!r}. "
            "owner_agent must be a non-empty string."
        )
    if not isinstance(user_id, str) or not user_id:
        raise ValueError(
            f"Invalid user_id: {user_id!r}. "
            "user_id must be a non-empty string."
        )

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
