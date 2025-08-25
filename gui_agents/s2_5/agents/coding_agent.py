import logging
import re
import textwrap
from typing import Dict, List, Tuple
import base64
import sys

from gui_agents.s2_5.agents.grounding import ACI
from gui_agents.s2_5.core.module import BaseModule
from gui_agents.s2_5.utils.common_utils import call_llm_safe
try:
    from coact.coding_agent import CODER_SYSTEM_MESSAGE  # may import desktop_env transitively
except Exception:
    CODER_SYSTEM_MESSAGE = (
        "- You are a programmer. Solve the task by writing code.\n"
        "- You can write code in ```bash``` or ```python``` blocks.\n"
        "- Wrap all your code in ONE code block and identify the language.\n"
        "- Verify results. Print progressive and final outputs.\n"
        "- If errors occur, analyze and try to fix them."
    )

logger = logging.getLogger("desktopenv.agent")


def _extract_first_code_block(text: str) -> Tuple[str, str]:
    """Return (language, code) for the first fenced code block in text, else ("", "")."""
    # Match ```lang\n...\n``` greedily minimal
    m = re.search(r"```([A-Za-z0-9_+-]*)\n([\s\S]*?)```", text)
    if not m:
        return "", ""
    lang = (m.group(1) or "").strip().lower()
    code = m.group(2)
    return lang, code


class CodingAgent(BaseModule):
    """Agent that generates Python/Bash code using coact's prompting and executes it safely."""

    def __init__(
        self,
        engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        super().__init__(engine_params, platform)
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.temperature = engine_params.get("temperature", 0.0)
        self.reset()

    def reset(self):
        # Single agent with coder system prompt
        sys_prompt = CODER_SYSTEM_MESSAGE.replace("{CLIENT_PASSWORD}", "")
        self.coder = self._create_agent(sys_prompt)
        self.turn_count = 0

    def _build_user_prompt(self, instruction: str) -> str:
        return textwrap.dedent(
            f"""
            # Task
            {instruction}

            # Instructions
            - Return ONE code block only, with the correct language tag: python or bash.
            - Prefer Python for data processing; prefer Bash for simple shell ops.
            - Print intermediate and final results.
            - Do not write files unless necessary. If writing files, explain in comments.
            """
        ).strip()

    def _action_to_run_subprocess(self, lang: str, code: str) -> str:
        """Return a Python snippet that, when exec'd, runs user code in a subprocess with timeout and prints output."""
        timeout_sec = 120
        safe_lang = "python" if lang in ("", "py", "python") else "bash"
        # Embed code via base64 to avoid quoting issues
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        if safe_lang == "bash":
            runner = f"""
import subprocess, base64
print("[coding-agent] Running bash code...\n", flush=True)
_code = base64.b64decode("{code_b64}").decode("utf-8")
proc = subprocess.run(_code, shell=True, capture_output=True, text=True, timeout={timeout_sec})
print(proc.stdout, end="")
if proc.stderr:
    print(proc.stderr, end="")
print("\n[coding-agent] Exit code:", proc.returncode, flush=True)
            """.strip()
        else:
            runner = f"""
import subprocess, sys, base64
print("[coding-agent] Running python code...\n", flush=True)
_code = base64.b64decode("{code_b64}").decode("utf-8")
proc = subprocess.run([sys.executable, "-c", _code], capture_output=True, text=True, timeout={timeout_sec})
print(proc.stdout, end="")
if proc.stderr:
    print(proc.stderr, end="")
print("\n[coding-agent] Exit code:", proc.returncode, flush=True)
            """.strip()
        return runner

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List[str]]:
        # Prepare input with optional screenshot
        self.coder.reset()
        user_prompt = self._build_user_prompt(instruction)
        if obs.get("screenshot"):
            self.coder.add_message(user_prompt, image_content=obs["screenshot"], role="user")
        else:
            self.coder.add_message(user_prompt, role="user")

        reply = call_llm_safe(self.coder, temperature=self.temperature)
        lang, code = _extract_first_code_block(reply)
        if not code.strip():
            logger.warning("CodingAgent: no code block detected; falling back to wait")
            return {"mode": "coding", "reply": reply}, ["agent.wait(2.0)"]

        action = self._action_to_run_subprocess(lang, code)
        info = {
            "mode": "coding",
            "language": lang or "python",
            "reply": reply,
        }
        self.turn_count += 1
        return info, [action]