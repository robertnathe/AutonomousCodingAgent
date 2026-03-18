import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from groq import Groq

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

@dataclass
class AgentConfig:
    model_id: str = "llama-3.3-70b-versatile"
    backup_dir: str = ".agent_backups"
    memory_db_path: str = "./agent_memory_db"
    memory_collection_name: str = "coding_agent_memory_v3"
    max_execution_timeout: int = 30
    max_code_timeout: int = 15
    max_install_timeout: int = 60
    
class MemoryManager:
    def __init__(self, config: AgentConfig):
        self.enabled = HAS_CHROMA
        self.collection = None
        if self.enabled:
            try:
                client = chromadb.PersistentClient(path=config.memory_db_path)
                ef = embedding_functions.DefaultEmbeddingFunction()
                self.collection = client.get_or_create_collection(
                    name=config.memory_collection_name,
                    embedding_function=ef,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize ChromaDB: {e}")
                self.enabled = False
    def retrieve(self, query: str, n_results: int = 3) -> str:
        if not self.enabled or not self.collection:
            return "Memory storage not available (chromadb not installed)."
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            if not results.get("documents") or not results["documents"][0]:
                return "No prior relevant history."
            return "\n".join(results["documents"][0])
        except Exception as e:
            return f"Memory retrieval error: {e}"
    def store(self, task: str, result: str):
        if not self.enabled or not self.collection:
            return
        try:
            count = self.collection.count()
            doc = f"Task: {task}\nResult: {result[-1000:]}"
            self.collection.add(
                documents=[doc],
                ids=[f"ep_{count}_{int(time.time())}"],
            )
        except Exception as e:
            print(f"Warning: Failed to store memory: {e}")

class FileManager:
    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.last_written_file: Optional[str] = None    
    def backup_file(self, file_path: str) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            return None
        existing = list(self.backup_dir.glob(f"{path.name}_v*.bak"))
        versions = []
        for backup in existing:
            match = re.search(r'_v(\d+)\.bak$', backup.name)
            if match:
                versions.append(int(match.group(1)))
        next_version = max(versions, default=0) + 1
        backup_path = self.backup_dir / f"{path.name}_v{next_version}.bak"
        try:
            shutil.copy2(path, backup_path)
            return str(backup_path)
        except Exception as e:
            print(f"Warning: Failed to backup {file_path}: {e}")
            return None
    def write_file(self, file_path: str, content: str) -> str:
        if not file_path or not isinstance(file_path, str) or not file_path.strip():
            return "CRITICAL ERROR: invalid file_path!"
        try:
            path = Path(file_path)
            self.backup_file(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if path.suffix.lower() in {'.py', '.python'}:
                self.last_written_file = file_path
            return f"Successfully wrote {file_path}."
        except Exception as e:
            return f"Error writing {file_path}: {e}"
    def read_file(self, file_path: str) -> str:
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"
    def undo_last_change(self, file_path: str) -> str:
        path = Path(file_path)
        existing = list(self.backup_dir.glob(f"{path.name}_v*.bak"))
        if not existing:
            return "No backup found."
        versions = []
        for backup in existing:
            match = re.search(r'_v(\d+)\.bak$', backup.name)
            if match:
                versions.append((int(match.group(1)), backup))
        if not versions:
            return "No backup found."
        latest_version, backup_path = max(versions, key=lambda x: x[0])
        try:
            shutil.copy2(backup_path, path)
            backup_path.unlink()
            return f"Reverted {file_path} to version {latest_version}."
        except Exception as e:
            return f"Error reverting file: {e}"

class CodeExecutor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.log_path = Path("./execution_traceback.log")
    def execute_file(self, file_path: str) -> str:
        try:
            self.log_path.write_text("")  # Clear log
            proc = subprocess.Popen(
                [sys.executable, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.config.max_execution_timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                rc = -9
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"=== {file_path} (rc={rc}) at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(stderr or "(no stderr)\n")
                f.write("=" * 60 + "\n\n")
            return (
                f"Return code: {rc}\n"
                f"STDOUT:\n{stdout.strip() or '(none)'}\n"
                f"STDERR (also written to {self.log_path}):\n{stderr.strip() or '(none)'}\n"
                f"Full traceback available via read_file('{self.log_path}')"
            )
        except Exception as e:
            return f"Execution failed: {e}"
    def execute_code(self, code: str) -> str:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.config.max_code_timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                rc = -9
            return (
                f"Return code: {rc}\n"
                f"STDOUT:\n{stdout.strip() or '(none)'}\n"
                f"STDERR:\n{stderr.strip() or '(none)'}"
            )
        except Exception as e:
            return f"Execution failed: {e}"
    def install_package(self, package: str) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=self.config.max_install_timeout
            )
            if result.returncode == 0:
                return f"Successfully installed {package}."
            else:
                return f"Failed to install {package}: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return f"Installation of {package} timed out."
        except Exception as e:
            return f"Error installing {package}: {e}"

class ToolCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    def get_key(self, tool_name: str, args: Dict) -> str:
        key_data = f"{tool_name}:{sorted(args.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    def set(self, key: str, value: Any):
        self._cache[key] = value
    def clear(self):
        self._cache.clear()

class CodingAgent:
    IGNORED_DIRS = {".git", "venv", "__pycache__", "node_modules", ".venv"}
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=api_key)
        self.file_manager = FileManager(self.config.backup_dir)
        self.executor = CodeExecutor(self.config)
        self.memory = MemoryManager(self.config)
        self.cache = ToolCache()
    def list_directory_files(self) -> List[str]:
        cache_key = self.cache.get_key("list_directory_files", {})
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        files = []
        ignored = self.IGNORED_DIRS | {self.config.backup_dir}
        for root, dirs, filenames in os.walk("."):
            dirs[:] = [d for d in dirs if d not in ignored]
            for filename in filenames:
                rel_path = os.path.relpath(os.path.join(root, filename), ".")
                files.append(rel_path)
        self.cache.set(cache_key, files)
        return files
    def read_file(self, file_path: str) -> str:
        cache_key = self.cache.get_key("read_file", {"file_path": file_path})
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        result = self.file_manager.read_file(file_path)
        self.cache.set(cache_key, result)
        return result
    def write_file(self, file_path: str, content: str) -> str:
        result = self.file_manager.write_file(file_path, content)
        self.cache.clear()
        return result
    def make_directory(self, dir_path: str) -> str:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.cache.clear()
            return f"Successfully created directory {dir_path}."
        except Exception as e:
            return f"Error creating directory {dir_path}: {e}"
    def undo_last_change(self, file_path: str) -> str:
        result = self.file_manager.undo_last_change(file_path)
        self.cache.clear()
        return result
    def execute_python_file(self, file_path: str) -> str:
        result = self.executor.execute_file(file_path)
        self.cache.clear()
        return result
    def run_python_code(self, code: str) -> str:
        return self.executor.execute_code(code)
    def install_package(self, package: str) -> str:
        return self.executor.install_package(package)
    def search_in_files(self, pattern: str, include_extensions: Optional[List[str]] = None) -> str:
        cache_key = self.cache.get_key(
            "search_in_files",
            {"pattern": pattern, "include_extensions": include_extensions or []}
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            cmd = ["grep", "-r", "-n", pattern, "."]
            if include_extensions:
                ext_pattern = "*.{" + ",".join(include_extensions) + "}"
                cmd.extend(["--include", ext_pattern])
            ignored = self.IGNORED_DIRS | {self.config.backup_dir}
            for ignore_dir in ignored:
                cmd.extend(["--exclude-dir", ignore_dir])
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                output = result.stdout.strip() or "No matches found."
            elif result.returncode == 1:
                output = "No matches found."
            else:
                output = f"Search failed: {result.stderr.strip()}"
        except Exception as e:
            output = f"Error during search: {e}"
        self.cache.set(cache_key, output)
        return output
    def get_environment_info(self) -> str:
        cache_key = self.cache.get_key("get_environment_info", {})
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            pip_out = subprocess.run(
                [sys.executable, "-m", "pip", "list"],
                capture_output=True,
                text=True,
                timeout=10
            ).stdout.strip()
            site_out = subprocess.run(
                [sys.executable, "-m", "site"],
                capture_output=True,
                text=True,
                timeout=10
            ).stdout.strip()
            result = (
                f"pip list:\n{pip_out or '(empty)'}\n\n"
                f"python -m site output:\n{site_out or '(empty)'}"
            )
        except Exception as e:
            result = f"Environment info failed: {e}"
        self.cache.set(cache_key, result)
        return result
    def clear_cache(self) -> str:
        self.cache.clear()
        return "Cache cleared."
    def _get_system_prompt(self) -> str:
        return (
            "You are an expert coding agent with full filesystem access. Work step-by-step. "
            "You have these tools:\n"
            "- list_directory_files ()\n"
            "- read_file(file_path: str)\n"
            "- write_file(file_path: str, content: str)\n"
            "- make_directory(dir_path: str)\n"
            "- execute_python_file(file_path: str)\n"
            "- undo_last_change(file_path: str)\n"
            "- run_python_code(code: str)\n"
            "- install_package(package: str)\n"
            "- search_in_files(pattern: str, include_extensions: list[str] = None)\n"
            "- get_environment_info ()\n"
            "- clear_cache ()\n\n"
            "STRICT OUTPUT FORMAT — FOLLOW EXACTLY:\n"
            "1. Step-by-step reasoning in plain text.\n"
            "2. **One or more** ```actions``` YAML blocks (use multiple for complex multi-step plans).\n"
            "3. **Exactly one** ```yaml``` outcome block at the very end.\n\n"
            "CRITICAL RULES FOR DATA TASKS (never ignore):\n"
            "• For cleaning dirty CSV/JSON data, ALWAYS use try/except around int() or float() conversion.\n"
            "• Never use only .isdigit() or truthy checks — they fail on decimals, negatives, 'abc', etc.\n"
            "• After writing cleaner.py, ALWAYS run it with execute_python_file, then use read_file on the output JSON to verify.\n"
            "• Only declare outcome: success after you have seen correct numeric results.\n\n"
            "Actions format example (no colon after ```actions):\n"
            "```actions\n"
            "- tool: write_file\n"
            "  args:\n"
            "    file_path: demo.py\n"
            "    content: |\n"
            "      print(42)\n"
            "```\n\n"
            "Outcome block (must be last):\n"
            "```yaml\n"
            "outcome: success\n"
            "```\n"
            "or\n"
            "```yaml\n"
            "outcome: continue\n"
            "```\n"
        )
    def _parse_and_run_actions(self, text: str) -> str:
        pattern = r"```actions\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if not matches:
            return "Observation: No actions block found."
        if len(matches) > 1:
            print(f"INFO: Executing {len(matches)} action blocks in sequence")
        all_observations = []
        for idx, yaml_text in enumerate(matches):
            block_num = idx + 1
            yaml_text = yaml_text.lstrip(":\n\t ").strip()
            try:
                actions = yaml.safe_load(yaml_text)
            except yaml.YAMLError as e:
                all_observations.append(f"Block {block_num}: Failed to parse YAML: {e}")
                continue
            if not isinstance(actions, list):
                all_observations.append(f"Block {block_num}: Actions should be a YAML list.")
                continue
            block_results = []
            for action in actions:
                if not isinstance(action, dict) or "tool" not in action or "args" not in action:
                    block_results.append("Observation: Malformed action (missing 'tool' or 'args')")
                    continue
                tool_name = action["tool"]
                args = action.get("args", {})
                tool = getattr(self, tool_name, None)
                if not tool:
                    block_results.append(f"Observation: Unknown tool '{tool_name}'")
                    continue
                try:
                    result = tool(**args)
                    block_results.append(f"Tool {tool_name} Result: {result}")
                except Exception as e:
                    block_results.append(f"Tool {tool_name} raised error: {e}")
            all_observations.append(f"--- ACTIONS BLOCK {block_num} ---\n" + "\n".join(block_results))
        return "\n\n".join(all_observations)
    def _check_outcome(self, text: str) -> bool:
        return bool(re.search(r"```yaml\s*outcome:\s*success\s*```", text, re.IGNORECASE))
    def _auto_verify_execution(self, observations: str) -> str:
        if "Tool execute_python_file Result:" in observations:
            return observations
        if self.file_manager.last_written_file:
            print(f"Auto-verification: Running {self.file_manager.last_written_file}")
            auto_obs = self.execute_python_file(self.file_manager.last_written_file)
            return f"{observations}\n--- Auto Verification ---\n{auto_obs}"
        return observations
    def run(self, task_description: str, max_turns: int = 3) -> str:
        print(f"Starting task: {task_description}\n")
        memory = self.memory.retrieve(task_description)
        session_context = []
        system_prompt = self._get_system_prompt()
        for turn in range(max_turns):
            print(f"\n{'='*60}")
            print(f"Turn {turn + 1}/{max_turns}")
            print(f"{'='*60}\n")
            full_prompt = (
                f"Long-term Memory:\n{memory}\n\n"
                f"Session History:\n" + "\n".join(session_context) + "\n\n"
                f"Task: {task_description}\n\n"
                "What do you do next?"
            )
            try:
                stream = self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    stream=True,
                )
                turn_text = ""
                print("--- Model Response ---")
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    turn_text += delta
                print("\n" + "-" * 60 + "\n")
            except Exception as e:
                print(f"Groq API error: {e}")
                return f"Agent failed due to API error: {e}"
            observations = self._parse_and_run_actions(turn_text)
            print("--- Observations ---")
            print(observations)
            print("-" * 60 + "\n")
            success_declared = self._check_outcome(turn_text)
            if success_declared:
                observations = self._auto_verify_execution(observations)
            session_context.append(
                f"Turn {turn+1} Output:\n{turn_text}\n\nObservation:\n{observations}"
            )
            if success_declared:
                print("✓ Task completed successfully.")
                self.memory.store(task_description, turn_text)
                return turn_text
        print("✗ Task failed to complete within turn limit.")
        self.memory.store(task_description, session_context[-1] if session_context else "No output")
        return "Task failed to complete within turn limit."

def main():
    agent = CodingAgent()
    
    combined_task = r"""
    Write a script pi_approx.py to approximate the value of $\pi$ using the Leibniz formula:
    $$\pi = 4 \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$$
    The agent must implement this as an iterative loop that continues until the absolute difference between the approximation 
    and math.pi is less than $10^{-5}$. Print the total number of iterations required to reach this level of precision.
    """
    result = agent.run(combined_task, max_turns=3)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result)
        
if __name__ == "__main__":
    main()
    
