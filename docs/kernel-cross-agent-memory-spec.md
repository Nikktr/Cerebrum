# Kernel Spec: Cross-Agent Shared Memory Retrieval

## Summary

The AIOS kernel's memory retrieval handler (`retrieve_memory` operation) must support two new optional parameters in the `params` dict: `user_id` and `sharing_policy`. These enable agents to discover memories created by other agents when those memories are explicitly marked as shared.

## Current Behavior

Today, when the kernel receives a `retrieve_memory` request:

```json
{
  "query_type": "memory",
  "agent_name": "assistant_agent",
  "query_data": {
    "query_class": "memory",
    "operation_type": "retrieve_memory",
    "params": {
      "content": "user profile preferences",
      "k": 5
    }
  }
}
```

The kernel scopes the search to memories owned by `agent_name` ("assistant_agent"). This means an agent can never see memories created by another agent, even if those memories were stored with `sharing_policy: "shared"` in their metadata.

## Required Behavior

The kernel must inspect `params` for two new optional keys and adjust its search scope accordingly:

| `user_id` present | `sharing_policy` present | Kernel behavior |
|---|---|---|
| No | No | **Existing behavior** — scope search to `agent_name` |
| Yes | No | Search across ALL agents' memories where metadata `user_id` matches the provided value |
| No | Yes | Search within `agent_name` scope, but filter results to memories where metadata `sharing_policy` matches |
| Yes | Yes | **Cross-agent search** — bypass agent scope, return memories matching BOTH `user_id` AND `sharing_policy` in metadata |

## Wire Format (New)

```json
{
  "query_type": "memory",
  "agent_name": "assistant_agent",
  "query_data": {
    "query_class": "memory",
    "operation_type": "retrieve_memory",
    "params": {
      "content": "user profile preferences",
      "k": 5,
      "user_id": "jane_doe",
      "sharing_policy": "shared"
    }
  }
}
```

When the kernel sees `user_id` + `sharing_policy` in params, it should:
1. Ignore the `agent_name` scope for search (don't restrict to assistant_agent's memories)
2. Search all memories where `metadata.user_id == "jane_doe"`
3. Filter to only those where `metadata.sharing_policy == "shared"`
4. Perform semantic search on `content` against the matching memory pool
5. Return top `k` results with full metadata in the response

## Expected Response Format

The response must include metadata for each search result (this may already be the case depending on provider):

```json
{
  "response_class": "memory",
  "search_results": [
    {
      "memory_id": "mem_abc123",
      "content": "{\"user_name\": \"Jane Doe\", \"preferred_tools\": [\"VS Code\", \"Docker\"], ...}",
      "score": 0.91,
      "metadata": {
        "owner_agent": "profile_agent",
        "user_id": "jane_doe",
        "memory_type": "profile",
        "sharing_policy": "shared"
      }
    },
    {
      "memory_id": "mem_def456",
      "content": "{\"current_project\": \"ML Pipeline\", \"goals\": [...], ...}",
      "score": 0.87,
      "metadata": {
        "owner_agent": "task_agent",
        "user_id": "jane_doe",
        "memory_type": "task_context",
        "sharing_policy": "shared"
      }
    }
  ],
  "success": true
}
```

## Implementation Guidance

### Where to Change

The memory retrieval handler that processes `operation_type == "retrieve_memory"`. This is where the kernel currently reads `params["content"]` and `params["k"]` and delegates to the memory provider.

### Logic to Add

```python
def handle_retrieve_memory(agent_name: str, params: dict):
    content = params["content"]
    k = params.get("k", 5)
    user_id = params.get("user_id")        # NEW — optional
    sharing_policy = params.get("sharing_policy")  # NEW — optional

    if user_id or sharing_policy:
        # Cross-agent or filtered search
        metadata_filter = {}
        if user_id:
            metadata_filter["user_id"] = user_id
        if sharing_policy:
            metadata_filter["sharing_policy"] = sharing_policy

        # Search across all memories matching the metadata filter
        # (bypass agent_name scoping when user_id is present)
        results = memory_provider.search(
            query=content,
            k=k,
            metadata_filter=metadata_filter,
            scope_agent=agent_name if not user_id else None,
        )
    else:
        # Existing behavior — agent-scoped search
        results = memory_provider.search(
            query=content,
            k=k,
            scope_agent=agent_name,
        )

    return results
```

### Provider-Specific Notes

**mem0:**
- mem0 already supports `user_id` as a scoping parameter in its search API
- You may need to pass `user_id` to mem0's search and add a post-filter for `sharing_policy` on the returned metadata

**zep:**
- zep supports `user_id` and `session_id` for scoping
- Similar approach: scope by `user_id`, post-filter by `sharing_policy`

**in-house:**
- Full control over the search implementation
- Can add metadata filtering directly to the vector search query

### Key Constraint

The metadata fields (`owner_agent`, `user_id`, `memory_type`, `sharing_policy`) are already being written by the SDK's `build_memory_metadata` helper at memory creation time. The kernel just needs to use them for filtering at retrieval time.

## Security Considerations

- `sharing_policy="shared"` is an explicit opt-in by the creating agent. The kernel should ONLY return cross-agent memories that have `sharing_policy="shared"` in their metadata.
- Memories with `sharing_policy="private"` (or no sharing_policy) must NEVER be returned to a different agent, even if `user_id` matches.
- When `user_id` is provided without `sharing_policy`, the kernel should still only return memories that are either owned by the requesting agent OR have `sharing_policy="shared"`.

## Validation Rules

- If `user_id` is present in params, it will be a non-empty string (SDK validates this)
- If `sharing_policy` is present, it will be either `"shared"` or `"private"` (SDK validates this)
- The kernel does not need to validate these values — the SDK handles it

## Auto-Inject Behavior

Auto-inject is the kernel feature that automatically injects relevant memories into LLM calls before they reach the model. It should follow the same sharing rules:

### Current Auto-Inject Behavior
- Kernel retrieves memories scoped to `agent_name` and injects them into the LLM context
- Only the agent's own memories are considered

### Required Auto-Inject Behavior
- When auto-inject is enabled, the kernel should also consider shared memories from other agents that match the same `user_id`
- The logic: retrieve memories where (`owner_agent == agent_name`) OR (`metadata.user_id == request_user_id` AND `metadata.sharing_policy == "shared"`)
- This means auto-inject naturally includes cross-agent shared context without the agent needing to call `search_memories` explicitly

### How user_id Flows to Auto-Inject
- The kernel already receives `agent_name` on every request
- For auto-inject to find cross-agent memories, it needs to know the `user_id` to scope by
- Option A: The kernel looks up the most recent `user_id` associated with this agent (from its stored memories)
- Option B: The SDK passes `user_id` as a top-level field in the request (alongside `agent_name`)
- **Recommended: Option A** — the kernel derives `user_id` from the agent's own memories, keeping the SDK interface simple

### Auto-Inject Filtering Rules
1. Agent's own memories (any sharing_policy) → always eligible for auto-inject
2. Other agents' memories with `sharing_policy="shared"` AND same `user_id` → eligible for auto-inject
3. Other agents' memories with `sharing_policy="private"` → NEVER auto-injected, regardless of user_id

### Benchmark Note
The benchmark harness sets `memory.auto_inject = false` to isolate the effect of explicit `search_memories` calls. This is correct for experimental control. In production, auto-inject + sharing rules work together to provide seamless personalization.

## Test Cases

1. **Agent-scoped (existing)**: No `user_id`/`sharing_policy` in params → only return memories where `owner_agent == agent_name`
2. **Cross-agent shared**: `user_id="jane"` + `sharing_policy="shared"` → return memories from ANY agent where `metadata.user_id == "jane"` AND `metadata.sharing_policy == "shared"`
3. **Cross-agent does NOT leak private**: Same as above but a memory has `sharing_policy="private"` → that memory must NOT appear in results
4. **User-scoped only**: `user_id="jane"` without `sharing_policy` → return memories where `metadata.user_id == "jane"` AND (`owner_agent == agent_name` OR `sharing_policy == "shared"`)
5. **Policy filter only**: `sharing_policy="shared"` without `user_id` → return memories where `owner_agent == agent_name` AND `metadata.sharing_policy == "shared"`
6. **Metadata in response**: All returned memories must include their full metadata dict in the response (owner_agent, user_id, memory_type, sharing_policy)
