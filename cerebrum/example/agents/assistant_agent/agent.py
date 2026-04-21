import os
import json

from cerebrum.llm.apis import llm_chat
from cerebrum.memory.apis import create_memory
from cerebrum.config.config_manager import config

from cerebrum.example.agents.shared_memory_utils import (
    build_memory_metadata,
    MEMORY_TYPE_CONVERSATION,
    POLICY_PRIVATE,
    POLICY_SHARED,
)

aios_kernel_url = config.get_kernel_url()


class AssistantAgent:
    """A personalized assistant agent that helps users with queries.

    Issues plain llm_chat calls. When the kernel's memory.auto_inject is
    enabled, shared context is injected automatically by the kernel.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.messages = []
        self.rounds = 0

    def load_config(self) -> dict:
        """Load agent configuration from config.json in the agent's directory."""
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, "config.json")

        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def run(self, task_input: str) -> dict:
        """Process user query and return a result dictionary.

        Args:
            task_input: The user's query string.

        Returns:
            A dict with agent_name, result, and rounds.
        """
        try:
            # Build system instruction from config description
            system_instruction = "".join(self.config.get("description", []))
            self.messages.append({"role": "system", "content": system_instruction})

            # Append user query
            self.messages.append({"role": "user", "content": task_input})

            # Call LLM — kernel auto_inject prepends shared context when enabled
            response = llm_chat(
                agent_name=self.agent_name,
                messages=self.messages,
                base_url=aios_kernel_url,
            )

            result_text = response["response"]["response_message"] if response else ""
            self.messages.append({"role": "assistant", "content": result_text})
            self.rounds += 1

            # Store conversation as memory
            try:
                self._store_conversation_memory(
                    user_id=getattr(self, 'user_id', self.agent_name),
                    content=f"User: {task_input}\nAssistant: {result_text}",
                )
            except Exception:
                pass  # Memory storage failure is non-critical

            return {
                "agent_name": self.agent_name,
                "result": result_text,
                "rounds": self.rounds,
            }

        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "result": f"Error: {e}",
                "rounds": self.rounds,
            }

    def _store_conversation_memory(self, user_id: str, content: str) -> None:
        """Store conversation turn as memory.

        Args:
            user_id: Identifier for the user this memory pertains to.
            content: The conversation content to store.
        """
        metadata = build_memory_metadata(
            owner_agent=self.agent_name,
            user_id=user_id,
            memory_type=MEMORY_TYPE_CONVERSATION,
            sharing_policy=POLICY_SHARED if getattr(self, 'share_memory', False) else POLICY_PRIVATE,
        )
        create_memory(
            agent_name=self.agent_name,
            content=content,
            metadata=metadata,
            base_url=aios_kernel_url,
        )
