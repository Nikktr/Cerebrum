import subprocess
import os
import sys

BRIDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)
BRIDGE_SRC = r"C:\AIOS\Cerebrum\cerebrum\example\agents"
if BRIDGE_SRC not in sys.path:
    sys.path.insert(0, BRIDGE_SRC)
from bridge_base import BridgeAgent


class CodexAgent(BridgeAgent):
    def __init__(self, agent_name):
        super().__init__(agent_name)
        self.codex_exe = self._find_codex()

    def _find_codex(self):
        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\hermes\node\codex.cmd"),
            os.path.expandvars(r"%APPDATA%\npm\codex.cmd"),
            "codex",
        ]
        for path in candidates:
            expanded = os.path.expandvars(path)
            if os.path.isfile(expanded):
                return expanded
        return "codex"

    def _run_codex(self, prompt, cwd=None):
        result = subprocess.run(
            [self.codex_exe, "exec", "--skip-git-repo-check", "-s", "workspace-write", "-"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
            shell=True,
            cwd=cwd,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += "\n[stderr]: " + result.stderr.strip()
        return output

    def run(self, task_input):
        project_id, project_path, task_text = self._resolve_project(task_input)
        context = self._get_shared_context(task_text, project_id)
        mcp_tools = self._get_mcp_tools(project_id)
        prompt = self._build_prompt_with_context(task_text, context, mcp_tools, project_path)
        cwd = project_path if project_path and os.path.isdir(project_path) else None

        try:
            output = self._run_codex(prompt, cwd)

            output, rounds = self._process_mcp_in_output(
                output, lambda p: self._run_codex(p, cwd)
            )

            result_text = output if output else "(no output)"
            self._save_to_memory(task_text, result_text, project_id)

            return {"agent_name": self.agent_name, "result": result_text, "rounds": rounds}

        except subprocess.TimeoutExpired:
            return {"agent_name": self.agent_name, "result": "Codex timed out after 600s.", "rounds": 1}
        except FileNotFoundError:
            return {"agent_name": self.agent_name, "result": f"Codex not found: {self.codex_exe}", "rounds": 1}
        except Exception as e:
            return {"agent_name": self.agent_name, "result": f"Error: {str(e)}", "rounds": 1}
