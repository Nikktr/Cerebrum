"""Property-based tests for shared_memory_utils using Hypothesis.

Feature: kernel-managed-shared-memory, Property 1: Metadata construction preserves all fields
Feature: kernel-managed-shared-memory, Property 2: Invalid enum values are rejected
"""

import re
import sys
sys.path.insert(0, ".")

import pytest

from hypothesis import given, settings, assume
from hypothesis.strategies import (
    text,
    sampled_from,
    dictionaries,
    one_of,
    integers,
    floats,
    booleans,
    none,
)

from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    FIELD_OWNER_AGENT,
    FIELD_USER_ID,
    FIELD_MEMORY_TYPE,
    FIELD_SHARING_POLICY,
)

VALID_MEMORY_TYPES = ["profile", "task_context", "conversation"]
VALID_SHARING_POLICIES = ["private", "shared"]
STANDARD_KEYS = {FIELD_OWNER_AGENT, FIELD_USER_ID, FIELD_MEMORY_TYPE, FIELD_SHARING_POLICY}

# Strategy for non-empty strings
non_empty_text = text(min_size=1)

# Strategy for extra kwargs values
extra_values = one_of(
    text(),
    integers(),
    floats(allow_nan=False),
    booleans(),
    none(),
)

# Strategy for extra kwargs keys that don't collide with standard fields
extra_keys = text(min_size=1).filter(lambda k: k not in STANDARD_KEYS)

# Strategy for extra kwargs dict
extra_kwargs = dictionaries(keys=extra_keys, values=extra_values, max_size=5)


# Feature: kernel-managed-shared-memory, Property 1: Metadata construction preserves all fields
class TestMetadataFieldPreservation:
    """**Validates: Requirements 1.1**"""

    @given(
        owner_agent=non_empty_text,
        user_id=non_empty_text,
        memory_type=sampled_from(VALID_MEMORY_TYPES),
        sharing_policy=sampled_from(VALID_SHARING_POLICIES),
        extra=extra_kwargs,
    )
    @settings(max_examples=100)
    def test_metadata_preserves_all_fields(
        self, owner_agent, user_id, memory_type, sharing_policy, extra
    ):
        """For any valid inputs, build_memory_metadata returns a dict with
        exactly the four standard keys plus extras, all with correct values."""
        result = build_memory_metadata(
            owner_agent=owner_agent,
            user_id=user_id,
            memory_type=memory_type,
            sharing_policy=sharing_policy,
            **extra,
        )

        # Standard fields have correct values
        assert result[FIELD_OWNER_AGENT] == owner_agent
        assert result[FIELD_USER_ID] == user_id
        assert result[FIELD_MEMORY_TYPE] == memory_type
        assert result[FIELD_SHARING_POLICY] == sharing_policy

        # Extra kwargs are preserved with correct values
        for key, value in extra.items():
            assert key in result
            assert result[key] == value

        # No unexpected keys: exactly standard + extras
        expected_keys = STANDARD_KEYS | set(extra.keys())
        assert set(result.keys()) == expected_keys


# Strategies for invalid enum values
invalid_sharing_policy = text(min_size=1).filter(
    lambda x: x not in set(VALID_SHARING_POLICIES)
)
invalid_memory_type = text(min_size=1).filter(
    lambda x: x not in set(VALID_MEMORY_TYPES)
)


# Feature: kernel-managed-shared-memory, Property 2: Invalid enum values are rejected
class TestInvalidEnumRejection:
    """**Validates: Requirements 1.3, 1.4**"""

    @given(
        invalid_policy=invalid_sharing_policy,
    )
    @settings(max_examples=100)
    def test_invalid_sharing_policy_raises(self, invalid_policy):
        """For any string not in {"private", "shared"}, build_memory_metadata
        raises ValueError with the invalid value in the message."""
        with pytest.raises(ValueError, match=re.escape(repr(invalid_policy))):
            build_memory_metadata(
                owner_agent="test_agent",
                user_id="test_user",
                memory_type="profile",
                sharing_policy=invalid_policy,
            )

    @given(
        invalid_type=invalid_memory_type,
    )
    @settings(max_examples=100)
    def test_invalid_memory_type_raises(self, invalid_type):
        """For any string not in {"profile", "task_context", "conversation"},
        build_memory_metadata raises ValueError with the invalid value in the message."""
        with pytest.raises(ValueError, match=re.escape(repr(invalid_type))):
            build_memory_metadata(
                owner_agent="test_agent",
                user_id="test_user",
                memory_type=invalid_type,
                sharing_policy="private",
            )


if __name__ == "__main__":
    test1 = TestMetadataFieldPreservation()
    print("Running Property 1: Metadata construction preserves all fields...")
    test1.test_metadata_preserves_all_fields()
    print("PASSED: Property 1")

    test2 = TestInvalidEnumRejection()
    print("Running Property 2: Invalid enum values are rejected...")
    test2.test_invalid_sharing_policy_raises()
    print("PASSED: Property 2 (invalid sharing_policy)")
    test2.test_invalid_memory_type_raises()
    print("PASSED: Property 2 (invalid memory_type)")
