# You are an advanced AI coding agent specializing in algorithmic discovery and optimization, powered by cutting-edge LLMs. 
# Your mission is to take an initial python algorithm, analyze its purpose, understand simulated evaluation results, and propose
# a single, significant improvement. Print the full corrected class or definition that are modified.
import re
import os
import time
import shutil
import subprocess
import threading
import tempfile
import math
import random
import requests
import traceback
import hashlib, json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache

@dataclass
class AgentConfig:
    backup_dir: str = ".agent_backups"
    memory_db_path: str = "./agent_memory_db"
    max_execution_timeout: int = 30
    max_code_timeout: int = 15
    max_install_timeout: int = 60
    semantic_similarity_threshold: float = 0.72
    max_memory_entries: int = 50
    # models: Dict[str, str] = field(default_factory=lambda: {
        # "groq": "qwen/qwen3-32b",
        # "openrouter": "cohere/north-mini-code:free",
        # "google": "gemini-3.5-flash",
        # "google_antigravity": "antigravity-preview-05-2026"
    # })
    # models: Dict[str, str] = field(default_factory=lambda: {
        # "groq": "openai/gpt-oss-120b",
        # "openrouter": "poolside/laguna-xs-2.1:free",
        # "google": "gemini-3.5-flash-lite",
        # "google_antigravity": "antigravity-preview-05-2026"
    # })
    models: Dict[str, str] = field(default_factory=lambda: {
        "groq": "llama-3.3-70b-versatile",
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
        "google": "gemini-3.1-flash-lite",
        "google_antigravity": "antigravity-preview-05-2026",
    })
    backend_priority: List[str] = field(default_factory=lambda: ["groq", "openrouter", "google"])
    global_rpm: int = 30
    max_actions_per_turn: int = 6
    evolution: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "auto_reflect": True,
        "max_context_turns": 6
    })
    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls()

def md5_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]

def get_time() -> float:
    return time.time()

STOP_WORDS = {
    "the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "by", "at", "an", "be", "this", "that",
    "it", "not", "or", "but", "are", "from", "has", "had", "have", "will", "would", "could", "should", "may", "can",
    "do", "does", "did", "was", "were", "been", "being", "am", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "what", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so",
    "than", "through", "too", "under", "until", "up", "very", "with", "would"
}

def tokenize(text: str) -> List[str]:
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    return [w for w in words if w not in STOP_WORDS]

def compute_tfidf_vector(text: str, idf: Dict[str, float], total_docs: int) -> List[Tuple[str, float]]:
    tokens = tokenize(text)
    freq: Dict[str, int] = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    if not freq:
        return []
    max_freq = max(freq.values())
    vec = []
    for term, count in freq.items():
        tf = count / max_freq
        idf_val = idf.get(term, math.log(1.0 + total_docs + 1) if total_docs > 0 else 1.0)
        vec.append((term, tf * idf_val))
    vec.sort(key=lambda x: x[0])
    return vec

def error_type_from_output(output: str) -> str:
    low = output.lower()
    if "no such file" in low:
        return "missing_file"
    if "name or service not known" in low or "name resolution" in low:
        return "name_resolution"
    if "segmentation fault" in low or "return code: 139" in low or "return code: -11" in low:
        return "segmentation_fault"
    if "return code: 1" in low:
        return "execution_error"
    if "compilation" in low or "g++" in low:
        return "compilation_error"
    return "unknown"

class RateLimiter:
    def __init__(self, rpm: int = 60):
        self.interval = 60.0 / rpm
        self.last_call = 0.0
        self.lock = threading.Lock()
    def acquire(self):
        with self.lock:
            now = time.time()
            sleep_time = self.last_call + self.interval - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.last_call = time.time()

class LLMClientBase:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: OrderedDict[str, Tuple[str, bool, str, float]] = OrderedDict()
        self._cache_max = 128
        self._cache_ttl = 3600.0
        self._lock = threading.Lock()

    def _make_cache_key(
        self,
        model: str,
        messages: List[Tuple[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        raw = json.dumps([model, messages, temperature, max_tokens], sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _fetch(
        self,
        model: str,
        messages: List[Tuple[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, bool, str]:
        raise NotImplementedError

    def chat(
        self,
        model: str,
        messages: List[Tuple[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stream: bool = False,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> Tuple[str, bool, str]:
        key = self._make_cache_key(model, messages, temperature, max_tokens)
        with self._lock:
            if key in self._cache:
                content, ok, err, timestamp = self._cache[key]
                if get_time() - timestamp <= self._cache_ttl:
                    self._cache.move_to_end(key)
                    return content, True, ""
                del self._cache[key]
        for attempt in range(max_attempts):
            try:
                content, ok, err = self._fetch(
                    model, messages, temperature, max_tokens
                )
                if ok:
                    with self._lock:
                        self._cache[key] = (content, True, "", get_time())
                        if len(self._cache) > self._cache_max:
                            self._cache.popitem(last=False)
                    return content, True, ""
                if attempt == max_attempts - 1:
                    return "", False, err
            except Exception as e:
                err = str(e)
                if attempt == max_attempts - 1:
                    return "", False, err
            time.sleep(
                min(60.0, base_delay * (2 ** attempt))
                * (0.5 + random.random())
            )
        return "", False, "Max retries exceeded"

class GroqClient(LLMClientBase):
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    def _fetch(self, model, messages, temperature, max_tokens):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": r, "content": c} for r, c in messages]
        }
        resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"], True, ""
        return "", False, f"HTTP {resp.status_code}: {resp.text}"

class OpenRouterClient(LLMClientBase):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    def __init__(self, api_key: str, rpm: int = 30):
        super().__init__(api_key)
        self.rate_limiter = RateLimiter(rpm)
    def _fetch(self, model, messages, temperature, max_tokens):
        self.rate_limiter.acquire()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "CodingAgent"
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": r, "content": c} for r, c in messages]
        }
        resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"], True, ""
        return "", False, f"HTTP {resp.status_code}: {resp.text}"

class GoogleClient(LLMClientBase):
    def _fetch(self, model: str, messages: List[Tuple[str, str]], temperature: float,
               max_tokens: int) -> Tuple[str, bool, str]:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={self.api_key}")
        contents = []
        system_instruction = None
        for role, text in messages:
            if role == "system":
                system_instruction = {"parts": [{"text": text}]}
            else:
                mapped_role = "user" if role == "user" else "model"
                if contents and contents[-1]["role"] == mapped_role:
                    contents[-1]["parts"][0]["text"] += "\n\n" + text
                else:
                    contents.append({"role": mapped_role, "parts": [{"text": text}]})
        if contents and contents[0]["role"] == "model":
            contents.insert(0, {"role": "user", "parts": [{"text": "Continue your work."}]})
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                if "candidates" in data and data["candidates"]:
                    try:
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        return content, True, ""
                    except (KeyError, IndexError):
                        return "", False, f"Unexpected response structure: {data}"
                return "", False, f"No candidates returned: {data}"
            return "", False, f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            return "", False, f"Network/Request Error: {str(e)}"

class CodeExecutor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.log_path = Path("./execution_traceback.log")
        self.last_compiled_exe: Optional[str] = None
        self.max_output_length = 4000
        self.auto_install_enabled = True
        self.resolved_dependencies = set()
        self.package_mapping = {
            "yaml": "PyYAML",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "bs4": "beautifulsoup4",
            "sklearn": "scikit-learn",
            "dotenv": "python-dotenv",
            "jwt": "PyJWT",
            "crypto": "pycryptodome",
        }

    def _truncate_output(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_output_length:
            return text
        half = self.max_output_length // 2
        omitted = len(text) - self.max_output_length
        return (
            text[:half]
            + f"\n... [TRUNCATED {omitted} CHARACTERS] ...\n"
            + text[-half:]
        )

    def _run(
        self,
        cmd: List[str],
        timeout: int,
        cwd: str = ".",
    ) -> Tuple[int, str, str]:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            return proc.returncode, out, err
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return -9, out, err

    def run_shell_command(self, command: str, timeout: int = 30) -> str:
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = proc.communicate(timeout=timeout)
            out = self._truncate_output(out)
            err = self._truncate_output(err)
            return (
                f"Return code: {proc.returncode}\n"
                f"STDOUT:\n{out}\n"
                f"STDERR:\n{err}"
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return f"Command timed out after {timeout} seconds."

    def _extract_missing_python_module(self, err: str) -> Optional[str]:
        match = re.search(
            r"ModuleNotFoundError: No module named '([^']+)'",
            err,
        )
        if match:
            return match.group(1)
        match = re.search(
            r"ImportError: cannot import name '.*?' from '([^']+)'",
            err,
        )
        if match:
            return match.group(1).split(".")[0]
        return None

    def execute_file(
        self,
        file_path: str,
        stdin: Optional[str] = None,
        retry_count: int = 0,
    ) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()
        exe_to_clean = None
        if not ext and self.last_compiled_exe and Path(self.last_compiled_exe).exists():
            cmd = [self.last_compiled_exe]
        elif ext == ".py":
            cmd = ["python3", str(path)]
        elif ext in {".cpp", ".cc", ".cxx"}:
            exe = self.compile_cpp(str(path))
            if "COMPILATION_FAILED:" in exe:
                return exe
            cmd = [exe]
            exe_to_clean = exe
        else:
            if path.exists() and os.access(path, os.X_OK):
                cmd = [str(path.resolve())]
            else:
                return f"Unsupported file type: {ext}"
        try:
            proc = subprocess.run(
                cmd,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=self.config.max_execution_timeout,
            )
            out = self._truncate_output(proc.stdout)
            err = self._truncate_output(proc.stderr)
            if proc.returncode != 0:
                if (
                    self.auto_install_enabled
                    and ext == ".py"
                    and retry_count < 2
                ):
                    missing_mod = self._extract_missing_python_module(err)
                    if (
                        missing_mod
                        and missing_mod not in self.resolved_dependencies
                    ):
                        install_name = self.package_mapping.get(
                            missing_mod,
                            missing_mod,
                        )
                        self.run_shell_command(f"pip install {install_name}")
                        self.resolved_dependencies.add(missing_mod)
                        return self.execute_file(
                            file_path,
                            stdin,
                            retry_count + 1,
                        )
                with open(self.log_path, "a") as f:
                    f.write(
                        f"--- FAILURE {file_path} RC={proc.returncode} ---\n"
                        f"STDERR:\n{err}\n"
                    )
                return (
                    f"Return code: {proc.returncode}\n"
                    f"STDOUT:\n{out}\n"
                    f"STDERR:\n{err}"
                )
            return f"Return code: 0\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        except subprocess.TimeoutExpired:
            return (
                f"Command timed out after "
                f"{self.config.max_execution_timeout} seconds."
            )
        except Exception as e:
            return f"Error: {e}"
        finally:
            if exe_to_clean and os.path.exists(exe_to_clean):
                try:
                    os.unlink(exe_to_clean)
                except Exception:
                    pass

    def execute_code(self, code: str, language: str = "auto") -> str:
        if language == "auto":
            language = (
                "cpp"
                if ("#include" in code or "int main(" in code)
                else "python"
            )
        tmp_file = None
        try:
            if language == "cpp":
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".cpp",
                    delete=False,
                ) as f:
                    f.write(code)
                    tmp_file = f.name
                return self.execute_file(tmp_file)
            elif language == "python":
                syntax = self.validate_syntax(code, "python")
                if "FAILED" in syntax:
                    return syntax + "\n(Execution skipped)"
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                ) as f:
                    f.write(code)
                    tmp_file = f.name
                return self.execute_file(tmp_file)
            return f"Unsupported language: {language}"
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass

    def compile_cpp(self, source_file: str, output_file: str = "") -> str:
        if not output_file:
            output_file = str(
                Path(tempfile.gettempdir())
                / f"{Path(source_file).stem}_{os.getpid()}.out"
            )
        compiler_flags = ["-std=c++17", "-O2"]
        linker_flags = []
        try:
            content = Path(source_file).read_text()
        except Exception as e:
            return f"COMPILATION_FAILED: Cannot read source: {e}"
        if "std::thread" in content:
            compiler_flags.append("-pthread")
            linker_flags.append("-pthread")
        if "<curl/curl.h>" in content:
            linker_flags.append("-lcurl")
        if "<openssl/" in content:
            linker_flags.extend(["-lcrypto", "-lssl"])
        if "<nlohmann/json.hpp>" in content:
            compiler_flags.append("-I/usr/include/nlohmann")
        cmd = (
            ["g++"]
            + compiler_flags
            + [source_file, "-o", output_file]
            + linker_flags
        )
        rc, _, stderr = self._run(cmd, 15)
        if rc != 0:
            return f"COMPILATION_FAILED: {self._truncate_output(stderr)}"
        self.last_compiled_exe = output_file
        return output_file

    def validate_syntax(self, code: str, language: str = "auto") -> str:
        if language == "auto":
            language = (
                "cpp"
                if ("#include" in code or "int main(" in code)
                else "python"
            )
        if language == "cpp":
            tmp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".cpp",
                    delete=False,
                ) as f:
                    f.write(code)
                    tmp_file = f.name
                rc, _, stderr = self._run(
                    ["g++", "-std=c++20", "-fsyntax-only", tmp_file],
                    10,
                )
                if rc == 0:
                    return "Syntax validation PASSED."
                return (
                    "Syntax validation FAILED: "
                    f"{self._truncate_output(stderr)}"
                )
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except Exception:
                        pass
        try:
            compile(code, "<string>", "exec")
            return "Syntax validation PASSED."
        except Exception as e:
            return f"Syntax validation FAILED: {e}"

@dataclass
class FailurePattern:
    task_hash: str
    error_type: str
    error_signature: str
    fix_actions: List[Dict[str, Any]]
    hint: str
    occurrence_count: int = 1
    last_seen: float = field(default_factory=get_time)

class SemanticMemoryEntry:
    __slots__ = (
        "task_hash",
        "task_text",
        "solution_code",
        "word_vector",
        "norm",
        "success_timestamp",
        "api_calls_saved",
        "access_count",
        "last_access_time",
        "term_set",
        "term_count",
        "is_plan",
        "success_count",
        "failure_count",
    )

    def __init__(self, task_hash, task_text, solution_code):
        self.task_hash = task_hash
        self.task_text = task_text
        self.solution_code = solution_code
        self.word_vector: List[Tuple[str, float]] = []
        self.norm = 0.0
        self.success_timestamp = get_time()
        self.api_calls_saved = 0
        self.access_count = 0
        self.last_access_time = 0.0
        self.term_set: set = set()
        self.term_count = 0
        self.is_plan = False
        self.success_count = 0
        self.failure_count = 0

class SemanticMemoryManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memory: List[SemanticMemoryEntry] = []
        self.exact: Dict[str, SemanticMemoryEntry] = {}
        self.inverted: Dict[str, List[Tuple[SemanticMemoryEntry, float]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.total_docs = 0
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "false_positive_avoided": 0,
            "total_retrieval_time_ms": 0.0,
            "retrieval_count": 0,
        }
        self.dynamic_thresholds: Dict[str, float] = {}
        self.max_entries = config.max_memory_entries
        self.db_path = Path(config.memory_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.failure_patterns: List[FailurePattern] = []
        self.error_index: Dict[str, List[FailurePattern]] = defaultdict(list)
        self._last_retrieval: Dict[str, Tuple[float, float, str]] = {}
        self.weights = [0.6, 0.4, 0.0]
        self.learning_rate = 0.02
        self.load_all()

    def load_all(self):
        self._load_memory()
        self._load_failures()

    def _load_memory(self):
        mem_file = self.db_path / "semantic_memory.json"
        if not mem_file.exists():
            return
        try:
            data = json.loads(mem_file.read_text())
            for item in data:
                entry = SemanticMemoryEntry(
                    item["task_hash"],
                    item["task_text"],
                    item["solution_code"],
                )
                entry.success_timestamp = item.get("timestamp", 0)
                entry.api_calls_saved = item.get("api_calls_saved", 0)
                entry.access_count = item.get("access_count", 0)
                entry.last_access_time = item.get("last_access_time", 0)
                entry.is_plan = item.get("is_plan", False)
                entry.success_count = max(
                    0,
                    int(item.get("success_count", 0)),
                )
                entry.failure_count = max(
                    0,
                    int(item.get("failure_count", 0)),
                )
                self.memory.append(entry)
                self.exact[entry.task_hash] = entry
            self._update_idf()
        except Exception as e:
            print(f"[Memory] Load error: {e}")

    def _load_failures(self):
        fail_file = self.db_path / "failure_patterns.json"
        if not fail_file.exists():
            return
        try:
            data = json.loads(fail_file.read_text())
            for item in data:
                pat = FailurePattern(**item)
                self.failure_patterns.append(pat)
                self.error_index[pat.error_type].append(pat)
        except Exception as e:
            print(f"[FailureMemory] Load error: {e}")

    def save_all(self):
        self._save_memory()
        self._save_failures()

    def _save_memory(self):
        try:
            data = [
                {
                    "task_hash": e.task_hash,
                    "task_text": e.task_text,
                    "solution_code": e.solution_code,
                    "timestamp": e.success_timestamp,
                    "api_calls_saved": e.api_calls_saved,
                    "access_count": e.access_count,
                    "last_access_time": e.last_access_time,
                    "vector": dict(e.word_vector),
                    "terms": list(e.term_set),
                    "norm": e.norm,
                    "is_plan": e.is_plan,
                    "success_count": e.success_count,
                    "failure_count": e.failure_count,
                }
                for e in reversed(self.memory)
            ]
            (
                self.db_path / "semantic_memory.json"
            ).write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Memory] Save error: {e}")

    def _save_failures(self):
        try:
            data = [
                {
                    "task_hash": p.task_hash,
                    "error_type": p.error_type,
                    "error_signature": p.error_signature,
                    "fix_actions": p.fix_actions,
                    "hint": p.hint,
                    "occurrence_count": p.occurrence_count,
                    "last_seen": p.last_seen,
                }
                for p in self.failure_patterns
            ]
            (
                self.db_path / "failure_patterns.json"
            ).write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[FailureMemory] Save error: {e}")

    def _update_idf(self):
        self.total_docs = len(self.memory)
        doc_freq = defaultdict(int)
        raw_tfs = {}
        for e in self.memory:
            tokens = tokenize(e.task_text)
            freq = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            raw_tfs[e.task_hash] = freq
            for term in freq.keys():
                doc_freq[term] += 1
        self.idf = {
            term: math.log(1.0 + self.total_docs / df)
            for term, df in doc_freq.items()
        }
        self.inverted.clear()
        for e in self.memory:
            freq = raw_tfs[e.task_hash]
            if not freq:
                e.word_vector = []
                e.term_set = set()
                e.term_count = 0
                e.norm = 0.0
                continue
            max_freq = max(freq.values())
            vec = []
            for term, count in freq.items():
                weight = (count / max_freq) * self.idf[term]
                vec.append((term, weight))
                self.inverted[term].append((e, weight))
            vec.sort(key=lambda x: x[0])
            e.word_vector = vec
            e.term_set = set(freq.keys())
            e.term_count = len(e.term_set)
            e.norm = sum(w * w for _, w in vec)

    def _compute_vector(self, text: str) -> List[Tuple[str, float]]:
        return compute_tfidf_vector(
            text,
            self.idf,
            self.total_docs,
        )

    def _memory_reliability(self, entry: SemanticMemoryEntry) -> float:
        total = entry.success_count + entry.failure_count
        if total <= 0:
            return 0.5
        return (
            entry.success_count + 1.0
        ) / (
            total + 2.0
        )

    def retrieve_similar(
        self,
        query: str,
        threshold: float = 0.75,
    ) -> Optional[str]:
        start = time.perf_counter()
        task_hash = md5_hash(query)
        if task_hash in self.exact:
            entry = self.exact[task_hash]
            now = get_time()
            entry.access_count += 1
            entry.last_access_time = now
            entry.api_calls_saved += 1
            self.save_all()
            self.stats["exact_hits"] += 1
            self.stats["retrieval_count"] += 1
            self.stats["total_retrieval_time_ms"] += (
                time.perf_counter() - start
            ) * 1000
            return entry.solution_code
        if not self.memory:
            self.stats["misses"] += 1
            return None
        query_vec = self._compute_vector(query)
        if not query_vec:
            self.stats["misses"] += 1
            return None
        query_norm = sum(w * w for _, w in query_vec)
        query_terms = {term for term, _ in query_vec}
        now = get_time()
        candidate_dot: Dict[str, float] = defaultdict(float)
        candidate_map: Dict[str, SemanticMemoryEntry] = {}
        for term, q_weight in query_vec:
            for entry, doc_weight in self.inverted.get(term, []):
                candidate_dot[entry.task_hash] += (
                    q_weight * doc_weight
                )
                candidate_map[entry.task_hash] = entry
        candidates_with_scores = []
        if candidate_dot:
            for h, dot in candidate_dot.items():
                entry = candidate_map[h]
                cos = (
                    dot /
                    (
                        math.sqrt(query_norm)
                        * math.sqrt(entry.norm)
                    )
                    if query_norm > 0 and entry.norm > 0
                    else 0.0
                )
                candidates_with_scores.append(
                    (entry, cos)
                )
            TOP_K = 32
            candidates_with_scores.sort(
                key=lambda x: x[1],
                reverse=True,
            )
            candidates_with_scores = candidates_with_scores[:TOP_K]
        else:
            candidates_with_scores = [
                (entry, 0.0)
                for entry in self.memory
            ]
        best_score = 0.0
        best_entry = None
        best_cos = 0.0
        best_jac = 0.0
        for entry, cos in candidates_with_scores:
            jac = self._jaccard(
                query_terms,
                entry.term_set,
            )
            semantic_score = self._hybrid(
                cos,
                jac,
            )
            age_days = (
                now - entry.success_timestamp
            ) / 86400.0
            freshness = (
                1.0
                - 0.1
                * min(
                    1.0,
                    max(
                        0.0,
                        (age_days - 1.0) / 6.0,
                    ),
                )
            )
            reliability = self._memory_reliability(entry)
            outcome_adjusted = (
                0.75 * semantic_score
                + 0.25 * reliability
            )
            score = outcome_adjusted * freshness
            if score > best_score:
                best_score = score
                best_entry = entry
                best_cos = cos
                best_jac = jac
        eff_thresh = self.dynamic_thresholds.get(
            task_hash,
            threshold,
        )
        if (
            best_entry is not None
            and best_score >= eff_thresh
        ):
            self.stats["semantic_hits"] += 1
            self.stats["retrieval_count"] += 1
            self.stats["total_retrieval_time_ms"] += (
                time.perf_counter() - start
            ) * 1000
            best_entry.access_count += 1
            best_entry.last_access_time = now
            best_entry.api_calls_saved += 1
            self.save_all()
            self._last_retrieval[task_hash] = (
                best_cos,
                best_jac,
                best_entry.task_hash,
            )
            return best_entry.solution_code
        self.stats["misses"] += 1
        return None

    def _jaccard(self, s1, s2):
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0

    def _hybrid(self, cos, jac):
        cos = max(
            0.0,
            min(1.0, float(cos)),
        )
        jac = max(
            0.0,
            min(1.0, float(jac)),
        )
        cos_weight = max(
            0.0,
            float(self.weights[0]),
        )
        jac_weight = max(
            0.0,
            float(self.weights[1]),
        )
        weight_sum = cos_weight + jac_weight
        if weight_sum <= 1e-12:
            return 0.5 * (cos + jac)
        score = (
            cos_weight * cos
            + jac_weight * jac
        ) / weight_sum
        return max(
            0.0,
            min(1.0, score),
        )

    def _get_candidates(self, query_vec):
        cand = {}
        for term, _ in query_vec:
            for entry, _ in self.inverted.get(term, []):
                cand[entry.task_hash] = entry
        return (
            list(cand.values())
            if cand
            else self.memory
        )

    def store(self, task: str, solution: str):
        task_hash = md5_hash(task)
        is_plan = False
        try:
            parsed = json.loads(solution)
            if (
                isinstance(parsed, list)
                and all("tool" in a for a in parsed)
            ):
                is_plan = True
        except Exception:
            pass
        if task_hash in self.exact:
            entry = self.exact[task_hash]
            entry.solution_code = solution
        else:
            entry = SemanticMemoryEntry(
                task_hash,
                task,
                solution,
            )
            self.memory.insert(0, entry)
            self.exact[task_hash] = entry
        entry.success_timestamp = get_time()
        entry.is_plan = is_plan
        entry.access_count += 1
        if len(self.memory) > self.max_entries:
            to_evict = min(
                self.memory,
                key=lambda e: (
                    0.65 * min(e.access_count, 10) / 10.0
                    + 0.35 * self._memory_reliability(e)
                ),
            )
            self.memory.remove(to_evict)
            del self.exact[to_evict.task_hash]
        self._update_idf()
        self.save_all()

    def store_failure(
        self,
        task_hash,
        error_type,
        error_output,
        fix_actions,
        hint,
    ):
        sig = (
            error_output[:200]
            .replace("\n", " ")
            .strip()
        )
        for pat in self.error_index.get(error_type, []):
            if (
                pat.error_signature in sig
                or sig in pat.error_signature
            ):
                pat.occurrence_count += 1
                pat.last_seen = get_time()
                if hint and len(hint) > len(pat.hint):
                    pat.hint = hint
                self.save_all()
                return
        pat = FailurePattern(
            task_hash=task_hash,
            error_type=error_type,
            error_signature=sig,
            fix_actions=fix_actions,
            hint=hint,
        )
        self.failure_patterns.append(pat)
        self.error_index[error_type].append(pat)
        self.save_all()

    def get_hint_for_error(
        self,
        task_hash,
        error_output,
    ):
        etype = error_type_from_output(error_output)
        sig = (
            error_output[:200]
            .replace("\n", " ")
            .strip()
        )
        for pat in self.error_index.get(etype, []):
            if (
                pat.error_signature in sig
                or sig in pat.error_signature
            ):
                pat.occurrence_count += 1
                pat.last_seen = get_time()
                self.save_all()
                return pat.hint
        return None

    def report_outcome(
        self,
        task_hash,
        success,
    ):
        retrieval = self._last_retrieval.pop(
            task_hash,
            None,
        )
        if retrieval is not None:
            cos, jac, entry_hash = retrieval
            entry = self.exact.get(entry_hash)
            if entry is not None:
                if success:
                    entry.success_count += 1
                else:
                    entry.failure_count += 1
            target = 1.0 if success else 0.0
            cos_weight = max(
                0.0,
                self.weights[0],
            )
            jac_weight = max(
                0.0,
                self.weights[1],
            )
            weight_sum = cos_weight + jac_weight
            if weight_sum > 1e-12:
                pred = (
                    cos_weight * cos
                    + jac_weight * jac
                ) / weight_sum
            else:
                pred = 0.5 * (cos + jac)
            pred = max(
                0.0,
                min(1.0, pred),
            )
            error = pred - target
            self.weights[0] -= (
                self.learning_rate
                * error
                * cos
            )
            self.weights[1] -= (
                self.learning_rate
                * error
                * jac
            )
            self.weights[2] *= 0.95
            for i in range(3):
                self.weights[i] = max(
                    -5.0,
                    min(5.0, self.weights[i]),
                )
            self.save_all()
        thresh = self.config.semantic_similarity_threshold
        if success:
            thresh = max(
                0.50,
                thresh * 0.95,
            )
        else:
            self._consec_fail = getattr(
                self,
                "_consec_fail",
                defaultdict(int),
            )
            self._consec_fail[task_hash] += 1

            if self._consec_fail[task_hash] >= 2:
                thresh = min(
                    0.90,
                    thresh * 1.05,
                )
                self._consec_fail[task_hash] = 0
        self.dynamic_thresholds[task_hash] = thresh

    def print_stats(self):
        print(
            f"Memory entries: "
            f"{len(self.memory)}/{self.max_entries}"
        )
        print(
            f"Hits exact={self.stats['exact_hits']} "
            f"semantic={self.stats['semantic_hits']} "
            f"misses={self.stats['misses']}"
        )
        print(
            "API calls saved: "
            f"{sum(e.api_calls_saved for e in self.memory)}"
        )
        
class ReflectionEngine:
    def __init__(self):
        self.patterns: Dict[str, int] = defaultdict(int)
        self.failure_log: List[Dict] = []
    def analyze(self, task: str, success: bool, observations: str, turn_count: int) -> Dict:
        analysis = {
            "timestamp": get_time(),
            "task_hash": md5_hash(task),
            "success": success,
            "turn_count": turn_count,
            "error_type": None,
            "bottleneck": None,
            "suggested_capability": None
        }
        if not success:
            etype = error_type_from_output(observations)
            analysis["error_type"] = etype
            if etype == "segmentation_fault":
                analysis["bottleneck"] = "null_pointer_or_buffer_overflow"
                analysis["suggested_capability"] = "memory_safety_checker"
            elif etype == "execution_error":
                analysis["suggested_capability"] = "execution_fixer"
            self.patterns[etype] += 1
            self.failure_log.append(analysis)
        return analysis

class ToolExecutor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.file_manager = FileManager(config.backup_dir)
        self.executor = CodeExecutor(config)
        self.ARG_ALIASES = {
            "write_file": {
                "path": "file_path",
                "filename": "file_path",
                "name": "file_path",
            },
            "read_file": {
                "path": "file_path",
                "file": "file_path",
            },
            "execute_file": {
                "path": "file_path",
                "file": "file_path",
            },
            "compile_cpp": {
                "path": "file_path",
                "file": "file_path",
            },
            "run_shell_command": {
                "cmd": "command",
                "shell_args": "command",
            },
            "make_directory": {
                "path": "dir_path",
                "dir_name": "dir_path",
                "name": "dir_path",
                "directory": "dir_path",
            },
        }
        self.tool_handlers = {
            "write_file": self._write_file,
            "read_file": self._read_file,
            "execute_file": self._execute_file,
            "make_directory": self._make_directory,
            "install_package": self._install_package,
            "compile_cpp": self._compile_cpp,
            "run_shell_command": self._run_shell_command,
            "finish": lambda **kw: "Task marked as finished. Verifying results...",
        }
        self.required_args = {
            "write_file": {"file_path", "content"},
            "read_file": {"file_path"},
            "execute_file": {"file_path"},
            "compile_cpp": {"file_path"},
            "run_shell_command": {"command"},
            "make_directory": {"dir_path"},
            "finish": set(),
        }

    def _write_file(self, file_path: str, content: str) -> str:
        return self.file_manager.write_file(file_path, content)

    def _read_file(self, file_path: str) -> str:
        return self.file_manager.read_file(file_path)

    def _execute_file(self, file_path: str, stdin: Optional[str] = None) -> str:
        return self.executor.execute_file(file_path, stdin)

    def _make_directory(self, dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        return f"Directory created: {dir_path}"

    def _install_package(self, package: str) -> str:
        return self.executor.run_shell_command(
            f"pip install {package}",
            timeout=self.config.max_install_timeout,
        )

    def _compile_cpp(self, file_path: str) -> str:
        return self.executor.compile_cpp(file_path)

    def _run_shell_command(self, command: str) -> str:
        return self.executor.run_shell_command(command)

    def _extract_json_blocks(self, text: str) -> List[str]:
        blocks = []
        idx = 0
        while idx < len(text):
            start = text.find("{", idx)
            if start == -1:
                break
            stack = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        stack += 1
                    elif ch == "}":
                        stack -= 1
                        if stack == 0:
                            blocks.append(text[start : i + 1])
                            idx = i
                            break
            else:
                break
            idx += 1
        return blocks

    def _smart_json_sanitize(self, text: str) -> str:
        result = []
        i = 0
        in_string = False
        escape_next = False
        while i < len(text):
            ch = text[i]
            if escape_next:
                result.append(ch)
                escape_next = False
                i += 1
                continue
            if ch == "\\":
                result.append(ch)
                escape_next = True
                i += 1
                continue
            if text[i : i + 3] == '"""':
                result.append('\\"\\"\\"')
                i += 3
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                i += 1
                continue
            if in_string:
                if ch == "\n":
                    result.append("\\n")
                elif ch == "\r":
                    result.append("\\r")
                elif ch == "\t":
                    result.append("\\t")
                else:
                    result.append(ch)
            else:
                result.append(ch)
            i += 1
        return "".join(result)

    def _loose_object_rescue(self, text: str) -> List[Dict[str, Any]]:
        objects = []
        clean = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()
        if clean.startswith("{") and clean.endswith("}"):
            return objects
        candidates = [
            clean,
            "{" + clean,
            clean + "}",
            "{" + clean + "}",
        ]
        for cand in candidates:
            cand = re.sub(r",(\s*[}\]])", r"\1", cand)
            cand = re.sub(
                r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
                r'\1"\2":',
                cand,
            )
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict) and "tool" in parsed:
                    objects.append(parsed)
                    return objects
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                import ast
                parsed = ast.literal_eval(cand)
                if isinstance(parsed, dict) and "tool" in parsed:
                    objects.append(parsed)
                    return objects
            except Exception:
                pass
        m = re.search(
            r'"tool"\s*:\s*"write_file".*?"file_path"\s*:\s*"([^"]+)".*?"content"\s*:\s*"((?:\\.|[^"\\])*)"',
            clean,
            re.DOTALL,
        )
        if m:
            objects.append({
                "tool": "write_file",
                "args": {
                    "file_path": m.group(1),
                    "content": m.group(2).replace("\\n", "\n").replace('\\"', '"'),
                },
            })
        return objects

    def _extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        objects = []
        clean_text = re.sub(
            r"```(?:json)?\s*(.*?)\s*```",
            r"\1",
            text,
            flags=re.DOTALL,
        )
        blocks = self._extract_json_blocks(clean_text)
        for block in blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "tool" in parsed:
                    objects.append(parsed)
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            sanitized = self._smart_json_sanitize(block)
            try:
                parsed = json.loads(sanitized)
                if isinstance(parsed, dict) and "tool" in parsed:
                    objects.append(parsed)
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            if '"tool"' in block or "'tool'" in block:
                repaired_objs = self._aggressive_json_repair(block)
                objects.extend(repaired_objs)

        if not objects:
            objects.extend(self._loose_object_rescue(clean_text))

        return objects

    def _aggressive_json_repair(self, text: str) -> List[Dict[str, Any]]:
        objects = []
        stack = []
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == "{":
                    stack.append(i)
                elif ch == "}":
                    if stack:
                        start = stack.pop()
                        if not stack:
                            candidate = text[start : i + 1]
                            if '"tool"' in candidate or "'tool'" in candidate:
                                repaired = self._repair_json_fragment(candidate)
                                try:
                                    obj = json.loads(repaired)
                                    if isinstance(obj, dict) and "tool" in obj:
                                        objects.append(obj)
                                except (json.JSONDecodeError, ValueError):
                                    import ast
                                    try:
                                        obj = ast.literal_eval(repaired)
                                        if isinstance(obj, dict) and "tool" in obj:
                                            objects.append(obj)
                                    except Exception:
                                        pass
        return objects

    def _repair_json_fragment(self, fragment: str) -> str:
        result = []
        in_string = False
        escape_next = False
        for ch in fragment:
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            if ch == "\\":
                result.append(ch)
                escape_next = True
                continue
            if ch == '"' and not in_string:
                in_string = True
                result.append(ch)
                continue
            if ch == '"' and in_string:
                result.append('"')
                in_string = False
                continue
            if in_string and ch in ("\n", "\r"):
                result.append("\\n" if ch == "\n" else "\\r")
                continue
            result.append(ch)
        repaired = "".join(result)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        repaired = re.sub(
            r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
            r'\1"\2":',
            repaired,
        )
        return repaired

    @staticmethod
    def _find_json_object_bounds(
        text: str,
        start_pos: int,
    ) -> Optional[Tuple[int, int]]:
        brace_start = text.rfind("{", 0, start_pos)
        if brace_start == -1:
            return None
        i = brace_start + 1
        stack = 1
        in_str = False
        escape = False
        while i < len(text) and stack > 0:
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    stack += 1
                elif ch == "}":
                    stack -= 1
            i += 1
        if stack == 0:
            return brace_start, i
        return None

    def _extract_write_file_direct(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        pattern = r'"tool"\s*:\s*"write_file"'
        for m in re.finditer(pattern, text):
            pos = m.start()
            bounds = self._find_json_object_bounds(text, pos)
            if bounds is None:
                continue
            obj_start, obj_end = bounds
            obj_str = text[obj_start:obj_end]
            fp_match = re.search(
                r'"file_path"\s*:\s*"((?:\\.|[^"\\])*)"',
                obj_str,
            )
            if not fp_match:
                continue
            file_path = (
                fp_match.group(1)
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t")
            )
            content_key = re.search(r'"content"\s*:\s*', obj_str)
            if not content_key:
                continue
            after_key = obj_str[content_key.end() :]
            if after_key.strip().startswith('"""'):
                triple_start = after_key.index('"""') + 3
                rest = after_key[triple_start:]
                close_match = re.search(r'"""\s*[,\}]', rest)
                if close_match:
                    content = rest[: close_match.start()]
                    content = (
                        content.replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace('\\"', '"')
                        .replace("\\\\", "\\")
                    )
                    return {
                        "tool": "write_file",
                        "args": {
                            "file_path": file_path,
                            "content": content,
                        },
                    }
            else:
                sq_match = re.match(
                    r'"((?:\\.|[^"\\])*)"',
                    after_key,
                    re.DOTALL,
                )
                if sq_match:
                    content = (
                        sq_match.group(1)
                        .replace('\\"', '"')
                        .replace("\\n", "\n")
                        .replace("\\t", "\t")
                    )
                    return {
                        "tool": "write_file",
                        "args": {
                            "file_path": file_path,
                            "content": content,
                        },
                    }
        return None

    def _extract_write_file_via_regex(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        pattern = r'\{\s*"tool"\s*:\s*"write_file"\s*,\s*"args"\s*:\s*\{'
        match = re.search(pattern, text)
        if not match:
            return None
        fp_match = re.search(
            r'"file_path"\s*:\s*"([^"]+)"',
            text[match.start() :],
        )
        if not fp_match:
            return None
        file_path = fp_match.group(1)
        content_start = text.find('"content"', match.start())
        if content_start == -1:
            return None
        triple_quote = text.find('"""', content_start)
        if triple_quote != -1:
            start = triple_quote + 3
            rest = text[start:]
            close_match = re.search(r'"""\s*\}', rest)
            if close_match:
                content = rest[: close_match.start()]
                content = (
                    content.replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
                return {
                    "tool": "write_file",
                    "args": {
                        "file_path": file_path,
                        "content": content,
                    },
                }
        sq_match = re.search(
            r'"content"\s*:\s*"((?:\\.|[^"\\])*)"',
            text[match.start() :],
        )
        if sq_match:
            content = (
                sq_match.group(1)
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
            return {
                "tool": "write_file",
                "args": {
                    "file_path": file_path,
                    "content": content,
                },
            }
        return None

    def _extract_xml_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        actions = []
        for match in re.finditer(
            r"<tool_call>(.*?)</tool_call>",
            text,
            re.DOTALL,
        ):
            block = match.group(1)
            tool_name_match = re.search(
                r"<tool_name>(.*?)</tool_name>",
                block,
                re.DOTALL,
            )
            if not tool_name_match:
                tool_name_fallback = re.sub(
                    r"<arg_key>.*",
                    "",
                    block,
                    flags=re.DOTALL,
                ).strip()
                if tool_name_fallback and "<" not in tool_name_fallback:
                    actions.append(
                        {
                            "tool": tool_name_fallback,
                            "args": {},
                        }
                    )
                continue
            tool_name = tool_name_match.group(1).strip()
            args = {}
            keys = re.findall(
                r"<arg_key>(.*?)</arg_key>",
                block,
                re.DOTALL,
            )
            vals = re.findall(
                r"<arg_value>(.*?)</arg_value>",
                block,
                re.DOTALL,
            )
            for k, v in zip(keys, vals):
                args[k.strip()] = v.strip()
            if tool_name:
                actions.append({"tool": tool_name, "args": args})
        return actions

    def _extract_actions(self, text: str) -> List[Dict[str, Any]]:
        objects = self._extract_json_objects(text)
        if not objects:
            write_file_action = self._extract_write_file_via_regex(text)
            if write_file_action:
                objects = [write_file_action]
            else:
                write_file_action = self._extract_write_file_direct(text)
                if write_file_action:
                    objects = [write_file_action]
        if not objects:
            objects = self._extract_xml_tool_calls(text)
        actions = []
        seen: set = set()
        for obj in objects:
            action = self._apply_aliases(obj)
            key = json.dumps(action, sort_keys=True)
            if key not in seen:
                seen.add(key)
                actions.append(action)
        return actions

    def _extract_action(self, text: str) -> Optional[Dict[str, Any]]:
        actions = self._extract_actions(text)
        return actions[0] if actions else None

    def _apply_aliases(self, action: Dict) -> Dict:
        tool = action.get("tool")
        if "args" not in action and tool:
            args = {k: v for k, v in action.items() if k != "tool"}
            if args:
                action["args"] = args
        if tool and tool in self.ARG_ALIASES:
            for bad, good in self.ARG_ALIASES[tool].items():
                if (
                    bad in action.get("args", {})
                    and good not in action.get("args", {})
                ):
                    action["args"][good] = action["args"].pop(bad)
        return action

    def _validate_content_syntax(
        self,
        file_path: str,
        content: str,
    ) -> Optional[str]:
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            result = self.executor.validate_syntax(content, "python")
            if "FAILED" in result:
                return result
        elif ext in {".cpp", ".cc", ".cxx"}:
            result = self.executor.validate_syntax(content, "cpp")
            if "FAILED" in result:
                return result
        return None

    def execute_action_dict(self, action: Dict[str, Any]) -> str:
        action = self._apply_aliases(action)
        tool = action.get("tool")
        args = action.get("args", {})
        if tool not in self.tool_handlers:
            return f"Error: Tool '{tool}' is not recognized."
        missing = self.required_args.get(tool, set()) - set(args.keys())
        if missing:
            return (
                f"Error: Missing required arguments for '{tool}': "
                f"{', '.join(missing)}"
            )
        if tool == "write_file":
            fp = args.get("file_path", "")
            content = args.get("content", "")
            syntax_err = self._validate_content_syntax(fp, content)
            if syntax_err:
                return (
                    f"Error: {syntax_err}\n"
                    "File was NOT written. Fix the syntax error and try again."
                )
        try:
            return self.tool_handlers[tool](**args)
        except Exception as e:
            return f"Exception executing tool '{tool}': {str(e)}"

    def parse_and_run_actions(self, text: str) -> str:
        actions = self._extract_actions(text)
        if not actions:
            return "Error: No valid JSON tool command found in your output."
        results = []
        for action in actions:
            results.append(self.execute_action_dict(action))
        return "\n---\n".join(results)



class LLMBackendManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.clients: Dict[str, LLMClientBase] = {}
        self.current_backend: Optional[str] = None
        self.cooldown_until: Dict[str, float] = {}
        self.rate_limiter = RateLimiter(config.global_rpm)
        self._abort = False
        self._init_clients()

    def _init_clients(self):
        groq_key = os.environ.get("GROQ_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        if groq_key:
            self.clients["groq"] = GroqClient(groq_key)
        if openrouter_key:
            self.clients["openrouter"] = OpenRouterClient(openrouter_key)
        if google_key:
            self.clients["google"] = GoogleClient(google_key)
        for backend in self.config.backend_priority:
            if backend in self.clients:
                self.current_backend = backend
                break
        if not self.current_backend:
            raise RuntimeError("No LLM backend available")

    def abort(self):
        self._abort = True

    def reset_abort(self):
        self._abort = False

    def reset_cooldowns(self):
        self.cooldown_until.clear()

    def _is_on_cooldown(self, backend: str) -> bool:
        if backend not in self.cooldown_until:
            return False
        if time.time() >= self.cooldown_until[backend]:
            del self.cooldown_until[backend]
            return False
        return True

    def _min_cooldown_remaining(self) -> float:
        now = time.time()
        remaining = float('inf')
        for expire_time in list(self.cooldown_until.values()):
            if expire_time > now:
                remaining = min(remaining, expire_time - now)
        return 0.0 if remaining == float('inf') else remaining

    def _set_cooldown(self, backend: str, err: str):
        low = err.lower()
        now = time.time()
        if "quota" in low and "tpd" in low:
            match = re.search(r"reset at (\d+)", low)
            wait = max(0, int(match.group(1)) - now) if match else 30
            self.cooldown_until[backend] = now + min(wait, 30)
        elif "429" in low:
            match = re.search(r"reset at (\d+)", low)
            wait = max(0, int(match.group(1)) - now) if match else 10
            self.cooldown_until[backend] = now + min(wait, 10)
        else:
            self.cooldown_until[backend] = now + 2

    def _clear_all_cooldowns(self):
        self.cooldown_until.clear()

    def chat(self, messages: List[Tuple[str, str]], temperature: float = 0.0,
             max_tokens: int = 2048) -> Tuple[str, bool, str]:
        if self._abort:
            return "", False, "Aborted by agent."
        self.rate_limiter.acquire()
        if self._abort:
            return "", False, "Aborted by agent."
        order = [self.current_backend] + [b for b in self.config.backend_priority
                                          if b != self.current_backend]
        for retry in range(3):
            tried_any = False
            for backend in order:
                if self._abort:
                    return "", False, "Aborted by agent."
                if self._is_on_cooldown(backend):
                    continue
                tried_any = True
                client = self.clients.get(backend)
                if not client:
                    continue
                model = self.config.models.get(backend, "unknown")
                content, ok, err = client.chat(model, messages, temperature, max_tokens)
                if ok:
                    self._clear_all_cooldowns()
                    if backend != self.current_backend:
                        print(f"[BACKEND] Switched to healthy backend: {backend}")
                    self.current_backend = backend
                    return content or "", True, ""
                self._set_cooldown(backend, err)
            if not tried_any and retry < 2:
                wait = self._min_cooldown_remaining()
                if wait <= 0:
                    continue
                print(f"[BACKEND] All backends on cooldown, waiting {wait:.1f}s for earliest cooldown expiry (retry {retry+1}/2)...")
                time.sleep(wait)
                continue
            elif retry < 2:
                sleep_time = min(15, 3 * (2 ** retry))
                print(f"[BACKEND] All backends exhausted, waiting {sleep_time}s (retry {retry+1}/2)...")
                time.sleep(sleep_time)
            else:
                break
        return "", False, "All backends exhausted."

    def _extract_json_with_bracket_matching(self, text: str) -> str:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx = text.find(start_char)
            if idx == -1:
                continue
            stack = 1
            in_str = False
            escape = False
            i = idx + 1
            while i < len(text) and stack > 0:
                ch = text[i]
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"' and not escape:
                    in_str = not in_str
                elif not in_str:
                    if ch == start_char:
                        stack += 1
                    elif ch == end_char:
                        stack -= 1
                i += 1
            if stack == 0:
                return text[idx:i]
        return ""

    def structured_chat(self, messages: List[Tuple[str, str]], schema: Dict[str, Any],
                        temperature: float = 0.0, max_tokens: int = 2048) -> Tuple[Dict, bool, str]:
        system = (
            "You are a precise JSON-only assistant. Output ONLY valid JSON matching the schema below.\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\nOutput the raw JSON immediately."
        )
        full = [("system", system)] + messages
        content, ok, err = self.chat(full, temperature, max_tokens)
        if not ok:
            return {}, False, err
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1)
        else:
            content = self._extract_json_with_bracket_matching(content)
        content = content.strip()
        if not content:
            return {}, False, "No JSON object found in response."
        for attempt_text in [content, self._smart_sanitize_for_structured(content)]:
            try:
                return json.loads(attempt_text), True, ""
            except Exception:
                pass
        return {}, False, f"JSON parse error: could not parse response (snippet: {content[:200]}...)"

    @staticmethod
    def _smart_sanitize_for_structured(text: str) -> str:
        result = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            if ch == '\\':
                result.append(ch)
                escape_next = True
                continue
            if ch == '"' and not in_string:
                in_string = True
                result.append(ch)
                continue
            if ch == '"' and in_string:
                in_string = False
                result.append(ch)
                continue
            if in_string and ch in ('\n', '\r'):
                result.append('\\n' if ch == '\n' else '\\r')
                continue
            result.append(ch)
        repaired = "".join(result)
        repaired = re.sub(r'(?<!\\)"""', r'\\"\\"\\"', repaired)
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        return repaired

    def get_current_backend(self) -> str:
        return self.current_backend or "unknown"

class TaskDecomposer:
    def __init__(self, llm: LLMBackendManager):
        self.llm = llm
    def decompose(self, task: str) -> List[str]:
        schema = {"checklist": ["sub-goal 1"]}
        prompt = f"Break this task into 3-5 sequential sub-goals. Output ONLY JSON.\nTask: {task}"
        res, ok, _ = self.llm.structured_chat([("user", prompt)], schema, temperature=0.0, max_tokens=256)
        if ok and isinstance(res, dict) and "checklist" in res:
            return res["checklist"]
        return [task]

class DependencyPlanner:
    def __init__(self, llm: LLMBackendManager):
        self.llm = llm
        self.schema = {
            "actions": [
                {
                    "tool": "write_file (or make_directory, etc.)",
                    "args": {"any key": "value"}
                }
            ]
        }
        self.tool_docs = {
            "write_file":       "args: file_path (str), content (str)",
            "read_file":        "args: file_path (str)",
            "execute_file":     "args: file_path (str), stdin (str, optional)",
            "compile_cpp":      "args: file_path (str)",
            "run_shell_command":"args: command (str)",
            "make_directory":   "args: dir_path (str)",
            "finish":           "args: (none)"
        }
    def generate_plan(self, task: str, expected_outputs: List[str]) -> List[Dict]:
        if not expected_outputs:
            return []
        tools_desc = "\n".join(f"- {t}: {doc}" for t, doc in self.tool_docs.items())
        prompt = (
            f"Task: {task}\n"
            f"Required output files: {', '.join(expected_outputs)}\n\n"
            f"Available tools and their arguments:\n{tools_desc}\n\n"
            "Create a JSON plan with a list of actions that will accomplish the task. "
            "Always create directories before writing files, and execute scripts that produce output files.\n"
            "Output ONLY the JSON array of actions."
        )
        plan, ok, _ = self.llm.structured_chat(
            [("user", prompt)], self.schema, temperature=0.0, max_tokens=1024
        )
        if ok and isinstance(plan, dict) and "actions" in plan and isinstance(plan["actions"], list):
            return self._fix_files(plan["actions"], expected_outputs)
        return self._fallback_plan(expected_outputs)
    def _fallback_plan(self, files):
        actions = []
        seen_dirs = set()
        for fp in files:
            parent = str(Path(fp).parent)
            if parent and parent != "." and parent not in seen_dirs:
                actions.append({"tool": "make_directory", "args": {"dir_path": parent}})
                seen_dirs.add(parent)
        return actions
    def _fix_files(self, actions, expected):
        names = {Path(p).name for p in expected}
        for act in actions:
            if act.get("tool") == "write_file":
                fp = act.get("args", {}).get("file_path")
                if fp and Path(fp).name in names:
                    for real in expected:
                        if Path(real).name == Path(fp).name:
                            act["args"]["file_path"] = real
                            break
        return actions

class TestCase:
    def __init__(self, id: str, name: str, desc: str, prompt: str,
                 expected_outputs: List[str], checks: List[Dict],
                 max_turns: int, timeout: int = 120):
        self.id = id
        self.name = name
        self.description = desc
        self.task_prompt = prompt
        self.expected_outputs = expected_outputs
        self.validation_checks = checks
        self.max_turns = max_turns
        self.timeout = timeout
        
class FileManager:
    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    def backup_file(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return ""
        pattern = re.compile(rf"{re.escape(path.name)}_v(\d+)\.bak")
        versions = [(int(m.group(1)), f) for f in self.backup_dir.iterdir()
                    if (m := pattern.match(f.name))]
        next_ver = max([v for v, _ in versions], default=0) + 1
        backup = self.backup_dir / f"{path.name}_v{next_ver}.bak"
        shutil.copy2(path, backup)
        return str(backup)
    def _fix_incorrect_includes(self, content: str) -> str:
        replacements = {
            r'#include\s*<json/json\.h>': '#include <nlohmann/json.hpp>',
            r'#include\s*"json/json\.h"': '#include <nlohmann/json.hpp>',
        }
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        return content
    def _inject_missing_includes(self, content: str) -> str:
        NEEDED = {
            r'\bstd::cout\b': '<iostream>',
            r'\bstd::string\b': '<string>',
            r'\bstd::vector\b': '<vector>',
            r'\bstd::thread\b': '<thread>',
            r'\bstd::mutex\b': '<mutex>',
            r'\bstd::ofstream\b': '<fstream>',
            r'\bstd::ifstream\b': '<fstream>',
            r'\bnlohmann::json\b': '<nlohmann/json.hpp>',
        }
        existing = set(re.findall(r'#include\s*[<"]([^>"]+)[>"]', content))
        to_add = set()
        for pat, inc in NEEDED.items():
            if re.search(pat, content) and inc not in existing:
                to_add.add(f'#include {inc}')
        if not to_add:
            return content
        lines = content.splitlines()
        last_include = max((i for i, l in enumerate(lines) if l.startswith("#include")), default=-1)
        pos = last_include + 1 if last_include >= 0 else 0
        for inc in sorted(to_add):
            lines.insert(pos, inc)
            pos += 1
        return "\n".join(lines)
    def write_file(self, file_path: str, content: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".cpp", ".cc", ".cxx"}:
            content = self._fix_incorrect_includes(content)
            content = self._inject_missing_includes(content)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        shutil.move(str(tmp), str(path))
        return f"File written successfully: {file_path}"
    def read_file(self, file_path: str) -> str:
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"
    def undo_last_change(self, file_path: str) -> str:
        path = Path(file_path)
        pattern = re.compile(rf"{re.escape(path.name)}_v(\d+)\.bak")
        versions = [(int(m.group(1)), f) for f in self.backup_dir.iterdir()
                    if (m := pattern.match(f.name))]
        if not versions:
            return "No backup found."
        latest = max(versions, key=lambda x: x[0])
        shutil.copy2(latest[1], path)
        latest[1].unlink()
        return f"Reverted {file_path} to version {latest[0]}."

class CodingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMBackendManager(config)
        self.tool_executor = ToolExecutor(config)
        self.semantic_memory = SemanticMemoryManager(config)
        self.reflection = ReflectionEngine()
        self.decomposer = TaskDecomposer(self.llm)
        self.planner = DependencyPlanner(self.llm)
        self.session_context = []
        self._last_validation_feedback = ""
        self._last_raw_error = ""
        self._written_files: List[str] = []
        self._expected_outputs: List[str] = []
        self._validation_checks: List[Dict] = []
        self._current_task: str = ""
        self._abort = False
        self._successful_plan_actions: List[Dict] = []

    def _system_prompt(self) -> str:
        return (
            "You are a Senior AI Coding Agent specializing in algorithmic optimization.\n"
            "Rules:\n"
            "1. Communicate exclusively through JSON tool calls.\n"
            "2. Format: {\"tool\": \"name\", \"args\": {\"arg\": \"val\"}}\n"
            "3. You MAY output multiple JSON actions in a single response, separated by newlines. "
            "Each will be executed in order.\n"
            "4. Available tools (use exactly these argument names):\n"
            "   - write_file: file_path (str), content (str)\n"
            "   - read_file: file_path (str)\n"
            "   - execute_file: file_path (str), stdin (str, optional)\n"
            "   - compile_cpp: file_path (str)\n"
            "   - run_shell_command: command (str)\n"
            "   - make_directory: dir_path (str)\n"
            "   - finish: (no args)\n"
            "5. For file creation tasks: write_file → execute_file → verify → finish.\n"
            "6. NEVER call finish until you have EXECUTED any script that creates output files.\n"
            "7. If validation fails, read the feedback and fix the exact issue mentioned.\n"
            "8. Always implement complete, functional code without placeholders or comments like '# TODO'.\n"
            "9. Pay close attention to VALIDATION REQUIREMENTS — your output must contain "
            "the exact verification strings requested (matching is case-insensitive).\n"
            "10. CRITICAL: Python strings that span multiple lines MUST use triple quotes "
            "(\"\"\"...\"\"\") or explicit \\n escapes. Single-quoted strings cannot contain literal newlines.\n"
            "11. CRITICAL: Inside JSON string values, escape all double quotes with backslash (\\\"). "
            "Never use bare unescaped triple quotes inside a JSON value.\n"
            "12. CRITICAL: Write files to the EXACT paths specified in EXPECTED OUTPUT FILES. "
            "Do NOT create extra subdirectories unless explicitly required.\n"
            "13. CRITICAL: For C++ JSON output, use ONLY <nlohmann/json.hpp>. Do NOT use <json/json.h>.\n"
            "14. CRITICAL: When validation checks require specific text in your script's output "
            "(e.g., 'Query successful', 'InsufficientFundsError', 'coordinates'), your code MUST "
            "explicitly print that EXACT text. Do not paraphrase or use alternative wording.\n"
            "15. CRITICAL: If a validation check provides stdin input, ensure your script reads from "
            "stdin (e.g., using input() or sys.stdin) and produces the required output strings.\n"
            "16. CRITICAL: Do NOT create unnecessary subdirectories like 'project/' or 'src/' unless "
            "the task explicitly requires them. Write all files to the root directory unless specified otherwise."
        )

    def _get_error_hint(self, error_output: str) -> str:
        low = error_output.lower()
        if "syntaxerror" in low and "unterminated string" in low:
            return ("HINT: Unterminated string literal. For multi-line strings use triple quotes "
                    "(\"\"\"...\"\"\") or \\n. Single-quoted f-strings cannot span lines.")
        if "indentationerror" in low:
            return "HINT: Python indentation error. Use exactly 4 spaces per level."
        if "modulenotfounderror" in low or "no module named" in low:
            return "HINT: Missing Python module. Use the standard library or install it first."
        if "no such file" in low:
            return "HINT: File not found. Ensure write_file succeeds before execute_file."
        if "compilation_error" in low or "COMPILATION_FAILED" in error_output:
            if "json/json.h" in error_output:
                return "HINT: C++ compilation failed. Replace <json/json.h> with <nlohmann/json.hpp>."
            return "HINT: C++ compilation failed. Ensure all required headers are included."
        return ""

    def _fix_action_paths(self, action: Dict[str, Any]) -> Dict[str, Any]:
        tool = action.get("tool")
        if tool not in ("write_file", "execute_file", "compile_cpp", "read_file"):
            return action
        fp = action.get("args", {}).get("file_path", "")
        if not fp or not self._expected_outputs:
            return action
        basename = Path(fp).name
        for expected in self._expected_outputs:
            if Path(expected).name == basename and fp != expected:
                print(f"[Agent] Redirecting {tool} path: {fp} -> {expected}")
                action["args"]["file_path"] = expected
                break
        return action

    def _build_progress_hint(self) -> str:
        if not self._expected_outputs and not self._validation_checks:
            return ""
        lines = ["📋 PROGRESS CHECKLIST:"]
        for fp in self._expected_outputs:
            exists = "✅" if Path(fp).exists() else "❌"
            lines.append(f"  {exists} {fp}")
        for check in self._validation_checks:
            if check.get("type") == "file_exists":
                exists = "✅" if Path(check["path"]).exists() else "❌"
                lines.append(f"  {exists} File exists: {check['path']}")
            elif check.get("type") == "execution":
                fp = check.get("file", "")
                exists = "✅" if Path(fp).exists() else "❌"
                lines.append(f"  {exists} Script ready: {fp}")
                if "expect" in check:
                    lines.append(f"     🔍 REQUIRED OUTPUT SUBSTRING: '{check['expect']}' (case-insensitive)")
                if "input" in check:
                    lines.append(f"     📥 STDIN INPUT: {repr(check['input'])}")
        missing = [fp for fp in self._expected_outputs if not Path(fp).exists()]
        if missing:
            lines.append(f"\n➡️  NEXT: Create these missing files: {', '.join(missing)}")
        else:
            lines.append("\n✅ All expected files exist. Execute validation scripts and call finish.")
        return "\n".join(lines)

    def _build_validation_requirements(self) -> str:
        if not self._validation_checks:
            return ""
        lines = ["\n🔒 VALIDATION REQUIREMENTS — YOUR CODE MUST SATISFY ALL OF THESE:"]
        for check in self._validation_checks:
            if check["type"] == "file_exists":
                lines.append(f"  • File MUST exist: {check['path']}")
            elif check["type"] == "execution":
                lines.append(f"  • Script {check['file']} MUST run with exit code 0")
                if "input" in check:
                    lines.append(f"    Stdin provided: {repr(check['input'])}")
                if "expect" in check:
                    lines.append(f"    Output MUST contain EXACTLY: '{check['expect']}'")
        return "\n".join(lines)

    def _build_solution_from_files(self) -> str:
        parts = []
        for fp in sorted(set(self._written_files)):
            if Path(fp).exists():
                try:
                    content = Path(fp).read_text(encoding="utf-8")
                    parts.append(f"=== {fp} ===\n{content}")
                except Exception:
                    pass
        return "\n\n".join(parts) if parts else ""

    def _execute_plan(self, plan: List[Dict]) -> List[str]:
        observations = []
        for i, action in enumerate(plan):
            action = self._fix_action_paths(action)
            obs = self.tool_executor.execute_action_dict(action)
            observations.append(obs)
            tool_name = action.get("tool", "unknown")
            self.session_context.append(f"Plan Step {i+1}: {tool_name} → {obs[:300]}")
            if tool_name == "write_file":
                fp = action.get("args", {}).get("file_path")
                if fp and fp not in self._written_files:
                    self._written_files.append(fp)
        return observations

    def _attempt_direct_execution(self, validation_checks: List[Dict]) -> None:
        if not validation_checks:
            return
        needed_files = set()
        for c in validation_checks:
            if c.get("type") == "file_exists":
                needed_files.add(c.get("path"))
            elif c.get("type") == "execution":
                needed_files.add(c.get("file"))
        missing = [f for f in needed_files if f and not Path(f).exists()]
        if not missing:
            return
        target = None
        target_stdin = None
        for fp in reversed(self._written_files):
            if not Path(fp).exists():
                continue
            if fp.endswith(".py"):
                target = fp
                for check in validation_checks:
                    if check.get("type") == "execution" and check.get("file") == fp:
                        target_stdin = check.get("input")
                        break
                break
            elif fp.endswith((".cpp", ".cc", ".cxx")):
                compiled = self.tool_executor.executor.compile_cpp(fp)
                if "COMPILATION_FAILED" not in compiled:
                    target = compiled
                    break
        if target:
            print(f"[Agent] 🔧 Auto-executing {target} to create missing files: {missing}")
            result = self.tool_executor.executor.execute_file(target, stdin=target_stdin)
            self.session_context.append(
                f"[Auto-exec] Executed {target} (missing: {missing})\nResult: {result[:400]}"
            )
            print(f"[Agent] Auto-exec result: {result[:200]}")

    def _check_early_success(self) -> Optional[str]:
        if not self.validator(self._expected_outputs, self._validation_checks):
            return None
        print("[Agent] ✅ Validation passed — terminating early.")
        solution_parts = {
            "plan_actions": self._successful_plan_actions,
            "file_contents": {}
        }
        for fp in sorted(set(self._written_files)):
            if Path(fp).exists():
                try:
                    solution_parts["file_contents"][fp] = Path(fp).read_text(encoding="utf-8")
                except Exception:
                    pass
        solution = json.dumps(solution_parts, indent=2)
        self.semantic_memory.store(self._current_task, solution)
        task_hash = md5_hash(self._current_task)
        self.semantic_memory.report_outcome(task_hash, success=True)
        self.reflection.analyze(self._current_task, success=True, observations="Validation successful", turn_count=len(self.session_context))
        return "SUCCESS: Task completed and validated."

    def validator(self, expected_outputs: List[str], validation_checks: List[Dict],
                  task_description: str = "") -> bool:
        for check in validation_checks:
            if check["type"] == "file_exists":
                if not Path(check["path"]).exists():
                    return False
            elif check["type"] == "execution":
                fp = check["file"]
                if not Path(fp).exists():
                    return False
                result = self.tool_executor.executor.execute_file(fp, stdin=check.get("input"))
                if "Return code: 0" not in result:
                    return False
                if "expect" in check and check["expect"].lower() not in result.lower():
                    return False
        for out_file in expected_outputs:
            if not Path(out_file).exists():
                return False
        return True

    def _is_critical_error(self, observation: str) -> bool:
        return (
            "Error: Syntax validation FAILED" in observation
            or "COMPILATION_FAILED" in observation
            or "Error: Tool" in observation
            or "Exception executing tool" in observation
        )

    def _extract_replayable_plan(self, cached_solution: str) -> Optional[List[Dict]]:
        try:
            parsed = json.loads(cached_solution)
            if isinstance(parsed, dict) and "plan_actions" in parsed:
                plan = parsed["plan_actions"]
                if isinstance(plan, list) and all("tool" in a for a in plan):
                    return plan
            if isinstance(parsed, list) and all("tool" in a for a in parsed):
                return parsed
        except Exception:
            pass
        return None

    def _score_plan_against_validation(self, plan: List[Dict], expected_outputs: List[str],
                                       validation_checks: List[Dict]) -> Tuple[float, List[str]]:
        if not plan:
            return 0.0, list(expected_outputs)
        written = set()
        for action in plan:
            if action.get("tool") == "write_file":
                fp = action.get("args", {}).get("file_path")
                if fp:
                    written.add(fp)
        missing = [f for f in expected_outputs if f not in written and not Path(f).exists()]
        for c in validation_checks:
            if c.get("type") == "file_exists" and c["path"] not in written and not Path(c["path"]).exists():
                missing.append(c["path"])
            elif c.get("type") == "execution" and c.get("file") not in written and not Path(c.get("file", "")).exists():
                missing.append(c["file"])
        total = max(1, len(expected_outputs) + len(validation_checks))
        score = 1.0 - (len(missing) / total)
        return score, missing

    def _repair_plan_suffix(self, partial_plan: List[Dict], missing: List[str],
                            task: str) -> List[Dict]:
        if not missing:
            return partial_plan
        schema = {
            "actions": [
                {"tool": "write_file (or execute_file / finish / ...)", "args": {"any": "value"}}
            ]
        }
        prompt = (
            f"Task: {task}\n"
            f"The following files / checks are still missing: {missing}\n"
            f"Existing successful prefix (do NOT repeat these):\n"
            f"{json.dumps(partial_plan[-6:], indent=2)}\n\n"
            "Output ONLY a JSON list of the additional actions needed to create the missing items "
            "and reach a successful finish. Prefer the shortest correct suffix."
        )
        repaired, ok, _ = self.llm.structured_chat(
            [("user", prompt)], schema, temperature=0.0, max_tokens=1024
        )
        if ok and isinstance(repaired, dict) and "actions" in repaired:
            suffix = repaired["actions"]
            useful = []
            for a in suffix:
                fp = a.get("args", {}).get("file_path") or a.get("args", {}).get("file")
                if not fp or any(Path(m).name == Path(fp).name for m in missing) or a.get("tool") in ("finish", "execute_file"):
                    useful.append(a)
            return partial_plan + useful
        return partial_plan

    def run(self, task: str, max_turns: int = 6,
            expected_outputs: List[str] = None,
            validation_checks: List[Dict] = None) -> str:
        expected_outputs = expected_outputs or []
        validation_checks = validation_checks or []
        self._current_task = task
        self._expected_outputs = expected_outputs
        self._validation_checks = validation_checks
        self.session_context = []
        self._last_validation_feedback = ""
        self._last_raw_error = ""
        self._written_files = []
        self._successful_plan_actions = []
        self._abort = False
        task_hash = md5_hash(task)
        cached_solution = self.semantic_memory.retrieve_similar(task)
        best_plan = None
        best_score = -1.0
        if cached_solution:
            candidates = []
            plan = self._extract_replayable_plan(cached_solution)
            if plan:
                candidates.append(plan)
            for entry in self.semantic_memory.memory[:8]:
                p = self._extract_replayable_plan(entry.solution_code)
                if p:
                    candidates.append(p)
            for cand in candidates:
                score, missing = self._score_plan_against_validation(cand, expected_outputs, validation_checks)
                if score > best_score:
                    best_score = score
                    best_plan = cand
                    best_missing = missing
            if best_plan is not None and best_score >= 0.4:
                print(f"[Agent] Replaying best cached plan (score={best_score:.2f}) ...")
                if best_score < 0.95:
                    best_plan = self._repair_plan_suffix(best_plan, best_missing, task)
                self._successful_plan_actions = list(best_plan)
                self._execute_plan(best_plan)
                early = self._check_early_success()
                if early:
                    return early
            else:
                self.session_context.append("[Memory] No sufficiently useful cached plan found.")
        
        print("[Agent] Generating plan...")
        sub_goals = self.decomposer.decompose(task)
        if len(sub_goals) > 1:
            print(f"[Agent] Task decomposed into {len(sub_goals)} sub-goals.")
            self.session_context.append(f"Strategic Plan Breakdown: {', '.join(sub_goals)}")
        plan = self.planner.generate_plan(task, expected_outputs)
        if plan:
            self._successful_plan_actions = list(plan)
            plan_observations = self._execute_plan(plan)
            print(f"[Agent] Plan executed ({len(plan)} steps).")
            early = self._check_early_success()
            if early:
                return early  
        hints = []
        if expected_outputs:
            hints.append(f"EXPECTED OUTPUT FILES: {', '.join(expected_outputs)}")
        if validation_checks:
            expectations = []
            for c in validation_checks:
                if c.get("type") == "file_exists":
                    expectations.append(f"  - File must exist: {c['path']}")
                elif c.get("type") == "execution":
                    exp = f"  - Script {c['file']} must run successfully (exit code 0)"
                    if "expect" in c:
                        exp += f" and output must contain: '{c['expect']}'"
                    expectations.append(exp)
            if expectations:
                hints.append("VALIDATION REQUIREMENTS:")
                hints.extend(expectations)  
        file_exists_hint = (
            f"\n{'='*60}\n"
            f"⚠️  CRITICAL WORKFLOW: write_file → execute_file → verify → finish\n"
            f"⚠️  DO NOT call finish until all files exist and scripts run successfully!\n"
            + ("\n".join(hints) if hints else "")
            + f"\n{'='*60}\n"
        )
        for turn in range(max_turns):
            if self._abort:
                print("[Agent] ⚠️ Abort signal received, stopping early.")
                return "FAILED: Aborted due to timeout or external signal."
            if self._last_raw_error:
                hint = self.semantic_memory.get_hint_for_error(task_hash, self._last_raw_error)
                if hint:
                    self.session_context.append(f"[FailureHint] {hint}")
                local_hint = self._get_error_hint(self._last_raw_error)
                if local_hint:
                    self.session_context.append(local_hint)   
            progress = self._build_progress_hint()
            validation_reqs = self._build_validation_requirements()
            history = "\n".join(self.session_context[-8:])
            warning = (f"\n🚨 VALIDATION FAILED 🚨\n{self._last_validation_feedback}\n"
                       if self._last_validation_feedback else "")      
            user_msg = (
                f"TASK: {task}\n"
                f"{file_exists_hint}"
                f"{validation_reqs}\n"
                f"{progress}\n"
                f"{warning}"
                f"HISTORY:\n{history}\n"
                f"──────────────────────────────────────\n"
                f"Turn {turn+1}/{max_turns} - What is your NEXT action?"
            )
            print(f"\n[Turn {turn+1}] Querying LLM...")
            content, ok, err = self.llm.chat(
                [("system", self._system_prompt()), ("user", user_msg)],
                max_tokens=4096
            )
            if not ok:
                print(f"[Turn {turn+1}] ❌ LLM Error: {err}")
                self.session_context.append(f"Turn {turn+1} LLM Error: {err}")
                continue
            print(f"[Turn {turn+1}] LLM response (first 300 chars): {content[:300]}")
            actions = self.tool_executor._extract_actions(content)
            if not actions:
                print(f"[Turn {turn+1}] ⚠️  No valid action extracted")
                self.session_context.append(f"Turn {turn+1}: No valid action found in LLM output.")
                continue
            halted = False
            finish_called = False
            for action_idx, action in enumerate(actions):
                if self._abort:
                    return "FAILED: Aborted due to timeout or external signal."
                tool_name = action.get("tool", "unknown")
                print(f"[Turn {turn+1}] 🔧 Action {action_idx+1}/{len(actions)}: {tool_name}")
                action = self._fix_action_paths(action)
                observation = self.tool_executor.execute_action_dict(action)
                print(f"[Turn {turn+1}] 📤 Observation (first 200 chars): {observation[:200]}")
                self.session_context.append(
                    f"Turn {turn+1}.{action_idx+1}: {tool_name}\n→ {observation[:500]}"
                )
                self._successful_plan_actions.append(action)
                if self._is_critical_error(observation):
                    self._last_raw_error = observation
                    error_type = error_type_from_output(observation)
                    self.semantic_memory.store_failure(
                        task_hash, error_type, observation, [],
                        self._get_error_hint(observation)
                    )
                    analysis = self.reflection.analyze(task, False, observation, turn + 1)
                    if analysis.get("suggested_capability"):
                        hint_msg = f"[Reflection] Potential bottleneck: {analysis.get('bottleneck', 'issue')}. Suggested capability: {analysis['suggested_capability']}"
                        self.session_context.append(hint_msg)
                        print(f"[Turn {turn+1}] {hint_msg}")
                    print(f"[Turn {turn+1}] 🛑 Critical error, halting action batch.")
                    halted = True
                    break
                if "Return code:" in observation and "Return code: 0" not in observation:
                    self._last_raw_error = observation
                    error_type = error_type_from_output(observation)
                    self.semantic_memory.store_failure(
                        task_hash, error_type, observation, [],
                        self._get_error_hint(observation)
                    )
                    analysis = self.reflection.analyze(task, False, observation, turn + 1)
                    if analysis.get("suggested_capability"):
                        hint_msg = f"[Reflection] Potential bottleneck: {analysis.get('bottleneck', 'issue')}. Suggested capability: {analysis['suggested_capability']}"
                        self.session_context.append(hint_msg)
                        print(f"[Turn {turn+1}] {hint_msg}")
                if tool_name == "write_file":
                    fp = action.get("args", {}).get("file_path")
                    if fp and fp not in self._written_files:
                        self._written_files.append(fp)
                    if fp:
                        print(f"[Turn {turn+1}] 📝 Wrote: {fp}") 
                if tool_name == "finish":
                    finish_called = True
                    halted = True
                    break
            if finish_called:
                print(f"[Turn {turn+1}] 🏁 Finish called, attempting validation...")
                self._attempt_direct_execution(validation_checks)
            if not halted or finish_called:
                early = self._check_early_success()
                if early:
                    return early
            if finish_called:
                missing_files = [f for f in expected_outputs if not Path(f).exists()]
                missing_checks = []
                for check in validation_checks:
                    if check["type"] == "file_exists" and not Path(check["path"]).exists():
                        missing_checks.append(f"❌ File not found: {check['path']}")
                    elif check["type"] == "execution":
                        fp = check.get("file", "")
                        if not Path(fp).exists():
                            missing_checks.append(f"❌ Script not found: {fp}")
                        else:
                            result = self.tool_executor.executor.execute_file(
                                fp, stdin=check.get("input"))
                            if "Return code: 0" not in result:
                                missing_checks.append(
                                    f"❌ Script {fp} failed (RC ≠ 0): {result[:150]}")
                            elif "expect" in check and check["expect"].lower() not in result.lower():
                                missing_checks.append(
                                    f"❌ Script {fp} output missing '{check['expect']}': {result[:150]}")       
                parts = []
                if missing_files:
                    parts.append(f"Missing output files: {missing_files}")
                if missing_checks:
                    parts.append("\n".join(missing_checks))
                self._last_validation_feedback = (
                    "\n".join(parts) if parts
                    else "Unknown validation failure. Check all files exist and scripts run successfully."
                )
                print(f"[Turn {turn+1}] ❌ VALIDATION FAILED:\n{self._last_validation_feedback}")
                self.session_context.append(
                    f"Turn {turn+1}: finish FAILED. {self._last_validation_feedback}"
                ) 
        print(f"\n❌ MAX TURNS REACHED ({max_turns})")
        if self._last_validation_feedback or self._last_raw_error:
            error_type = error_type_from_output(self._last_validation_feedback or self._last_raw_error)
            self.semantic_memory.store_failure(
                task_hash, error_type,
                self._last_validation_feedback or self._last_raw_error, [],
                "Task failed after max turns. Review validation requirements."
            )
        self.semantic_memory.report_outcome(task_hash, success=False)
        self.reflection.analyze(task, False, self._last_validation_feedback or "Max turns reached", max_turns)
        return "FAILED: Maximum turns reached."

class TestSuite:
    def __init__(self):
        self.cases = self._default_cases()

    def _default_cases(self) -> List[TestCase]:
        cases = []
        cases.append(TestCase(
            "T1", "Automated Debugging via Log Analysis",
            "Automated Debugging via Log Analysis",
            ("The file memory_leak.cpp already exists and contains a segfault bug. "
             "1. Execute the file to observe the crash. 2. Read execution_traceback.log. "
             "3. Fix the bug so the file runs and exits with code 0. Do NOT simply replace it with unrelated code."),
            ["memory_leak.cpp"],
            [{"type": "execution", "file": "memory_leak.cpp"}],
            6, 120
        ))
        cases.append(TestCase(
            "T2", "Concurrent Hashing",
            "Parallel processing with concurrent.futures",
            ("Create concurrent_hash.py scanning for .txt/.py files, calculate SHA-256 in parallel. "
             "Store {filename: hash} dict. Print execution time and verify count matches."),
            ["concurrent_hash.py"],
            [{"type": "execution", "file": "concurrent_hash.py", "expect": "SHA-256"}],
            4, 120
        ))
        cases.append(TestCase(
            "T3", "Multi-Language Data Bridge",
            "Multi-Language Data Bridge",
            ("Create a C++ program named generator.cpp that simulates 10^6 particle interactions and saves "
             "the state to a data.json file using the nlohmann/json library. Subsequently, create a Python "
             "script analyzer.py that reads this JSON file, calculates the mean energy of the particles, and "
             "generates a summary report. This tests the agent ability to manage dependencies across different "
             "languages and ensure data consistency between processes."),
            ["generator.cpp", "analyzer.py"],
            [{"type": "execution", "file": "analyzer.py", "expect": "Mean"}],
            6, 120
        ))
        cases.append(TestCase(
            "T4", "Concurrent Data Processing with Thread Safety",
            "Concurrent Data Processing with Thread Safety",
            ("Create a CSV file named \"data.csv\" with 10,000 rows of random integers. Implement a C++ "
             "program \"data_processor.cpp\" that reads this file, processes it in parallel using threads, "
             "and outputs the processed results to \"output.json\". Ensure thread safety with appropriate "
             "synchronization mechanisms."),
            ["data_processor.cpp", "output.json"],
            [{"type": "execution", "file": "data_processor.cpp", "expect": "Processed"}],
            8, 120
        ))
        cases.append(TestCase(
            "T5", "Secure Input Validation",
            "Secure Input Validation",
            ("Create a single Python script named secure_query.py that demonstrates secure SQLite querying. "
             "The script MUST: (1) Create an SQLite database file and a 'users' table with at least one row "
             "(e.g., name='John Doe'), (2) Read lines from stdin in a loop until 'exit' is received, "
             "(3) Use parameterized queries to search for users by name, (4) Print 'Query successful' "
             "when a query executes without error, (5) Attempt a common SQL injection attack on itself "
             "and show the attack is blocked. The script will be executed with stdin 'John Doe\\nexit\\n' "
             "and the output MUST contain 'Query successful'."),
            ["secure_query.py"],
            [{"type": "execution", "file": "secure_query.py", "input": "John Doe\nexit\n", "expect": "Query successful"}],
            5, 120
        ))
        cases.append(TestCase(
            "T6", "Monte Carlo Integration",
            "Probabilistic numerical methods",
            ("Create a c++ program monte_carlo.cpp to estimate ∫sin(x)dx from 0 to π using 1,000,000 random points. "
             "Compare estimate to analytical value 2. Print absolute error."),
            ["monte_carlo.cpp"],
            [{"type": "execution", "file": "monte_carlo.cpp", "expect": "Absolute error"}],
            5, 120
        ))
        cases.append(TestCase(
            "T7", "Import Restructuring",
            "Handle directory moves and import updates",
            ("1. Create math_lib/operations.py with power(a,b) function. Create app.py importing it and "
             "printing 2**10. 2. Move operations.py to core/utils/, update imports, verify output still 1024."),
            ["app.py", "core/utils/operations.py"],
            [{"type": "execution", "file": "app.py", "expect": "1024"}],
            8, 120
        ))
        cases.append(TestCase(
            "T8", "Vending Machine OOP",
            "Custom exceptions and state management",
            ("Create vending.py with VendingMachine class and InsufficientFundsError exception. Demonstrate: "
             "deposit $2.00, try buying a $2.50 item, catch the InsufficientFundsError, and print the EXACT text "
             "'InsufficientFundsError' (this exact string is required for validation). Then deposit $1 more and "
             "buy successfully. Finally print the remaining balance and inventory."),
            ["vending.py"],
            [{"type": "execution", "file": "vending.py", "expect": "InsufficientFundsError"}],
            6, 120
        ))
        cases.append(TestCase(
            "T9", "Pi Approximation",
            "Numerical methods and convergence",
            ("Create pi_approx.py using Leibniz formula: π = 4 * Σ((-1)^n/(2n+1)). Iterate until "
             "|approximation - math.pi| < 10^-5. Print iterations needed."),
            ["pi_approx.py"],
            [{"type": "execution", "file": "pi_approx.py", "expect": "iterations"}],
            4, 120
        ))
        cases.append(TestCase(
            "T10", "Grid BFS Pathfinding",
            "Graph traversal and obstacle handling",
            ("Create grid_bfs.py with a 10x10 grid containing some obstacles. Implement BFS from (0,0) to (9,9). "
             "Print the path using the EXACT text 'coordinates' (for example: 'Path coordinates: [...]') and the step count. "
             "If no path exists, print 'No path found.'."),
            ["grid_bfs.py"],
            [{"type": "execution", "file": "grid_bfs.py", "expect": "coordinates"}],
            4, 120
        ))
        cases.append(TestCase(
            "T11", "SQLite Relational Query",
            "Complex JOINs and subqueries",
            (
                "Create SQLite database with Departments and Employees tables. Insert 3 departments, 10 employees. Query: \n"
                "find employees earning more than their dept average. Save to high_earners.json. Use proper foreign keys.\n"
            ),
            ["high_earners.json"],
            [{"type": "file_exists", "path": "high_earners.json"}],
            8, 120
        ))
        cases.append(TestCase(
            "T12", "Memoization Decorator",
            "Higher-order functions and performance",
            ("Create fibonacci_memoized.py with memoize decorator and apply to recursive Fibonacci. "
             "Calculate F(50) with timing. Compare to F(30) without decorator."),
            ["fibonacci_memoized.py"],
            [{"type": "execution", "file": "fibonacci_memoized.py", "expect": "F(50)"}],
            6, 120
        ))
        cases.append(TestCase(
            "T13", "Cross-File Refactoring",
            "Cross-File Refactoring",
            ("Task the agent with renaming a specific class or utility function in a multi-file project. "
             "For example, it must create a directory src/math, move several mathematical functions from a "
             "single utils.py into separate modules, and then update the import statements in an app.py "
             "located in the root directory. This evaluates the efficiency of the FileManager in handling "
             "path updates and recursive directory searches."),
            ["app.py", "utils.py"],
            [{"type": "execution", "file": "app.py", "expect": "power"}],
            4, 120
        ))
        cases.append(TestCase(
            "T14", "Data Cleaning Pipeline",
            "Handle dirty CSV data with type validation",
            ("Create raw_data.csv (Name, Age, Salary) with 6 rows including non-numeric ages/empty salaries. "
             "Create cleaner.py to filter invalid rows, calculate average salary, save to processed_data.json."),
            ["raw_data.csv", "cleaner.py", "processed_data.json"],
            [{"type": "file_exists", "path": "processed_data.json"}],
            6, 120
        ))
        cases.append(TestCase(
            "T15", "Networked Resource Fetching",
            "Networked Resource Fetching",
            ("Require the agent to use the curl capability to interface with a public API to retrieve "
             "current weather or financial data. The agent must then use openssl/md5 to create a unique "
             "checksum of the response to verify data integrity before processing. This evaluates the "
             "ability to handle external libraries and asynchronous-like behavior within the execution environment."),
            ["api_fetch.cpp"],
            [{"type": "execution", "file": "api_fetch.cpp", "expect": "Checksum"}],
            4, 120
        ))
        cases.append(TestCase(
            "T16", "LRU Cache with TTL",
            "Data structures and time-bound expiration",
            ("Implement an LRU (Least Recently Used) cache with TTL (Time-To-Live) in lru_cache.py. "
             "Requirements: O(1) get and put. Entries expire automatically after TTL seconds. "
             "Must support: get(key), put(key, value, ttl), size(). Print demonstration showing "
             "eviction due to capacity and expiration due to TTL."),
            ["lru_cache.py"],
            [
                {"type": "execution", "file": "lru_cache.py", "expect": "LRU"},
                {"type": "execution", "file": "lru_cache.py", "expect": "expired"},
                {"type": "execution", "file": "lru_cache.py", "expect": "evicted"},
                {"type": "execution", "file": "lru_cache.py", "expect": "O(1)"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T17", "Topological Sort with Cycle Detection",
            "Graph algorithms and error handling",
            ("Create topo_sort.py that reads a DAG from edges list and outputs a valid topological ordering. "
             "If a cycle is detected, raise CycleError with message 'Graph contains a cycle'. "
             "Use Kahn's algorithm (BFS-based). Test with: A->B, A->C, B->D, C->D."),
            ["topo_sort.py"],
            [
                {"type": "execution", "file": "topo_sort.py", "expect": "Topological"},
                {"type": "execution", "file": "topo_sort.py", "expect": "CycleError"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T18", "Mini Regex Engine",
            "Finite automata and string algorithms",
            ("Create regex_engine.py implementing a basic regex matcher supporting: . (any char), * (zero+ of prev), "
             "| (alternation), and literal characters. Do NOT use re module. Implement match(pattern, text) returning bool. "
             "Demonstrate: match('a*b', 'aaab')==True, match('a.b', 'acb')==True, match('a|b', 'b')==True."),
            ["regex_engine.py"],
            [
                {"type": "execution", "file": "regex_engine.py", "expect": "True"},
                {"type": "execution", "file": "regex_engine.py", "expect": "False"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T19", "Producer-Consumer with Backpressure",
            "Advanced concurrency and queue management",
            ("Create producer_consumer.py with bounded queue (maxsize=5). Producer generates 20 items. "
             "Consumer processes items. If queue full, producer blocks. If queue empty, consumer blocks. "
             "Use threading.Condition for synchronization. Print 'produced X' and 'consumed X' for each item. "
             "Final output must contain 'Done: 20 produced, 20 consumed'."),
            ["producer_consumer.py"],
            [
                {"type": "execution", "file": "producer_consumer.py",
                 "expect": "Done: 20 produced, 20 consumed"},
                {"type": "execution", "file": "producer_consumer.py", "expect": "produced"},
                {"type": "execution", "file": "producer_consumer.py", "expect": "consumed"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T20", "JSON Parser from Scratch",
            "Recursive descent parsing and tokenization",
            ("Create json_parser.py that parses a JSON string WITHOUT using json module. Support: objects, arrays, "
             "strings, numbers, booleans, null. Return native Python dict/list/str/int/float/None. "
             "Demonstrate parsing '{\"name\": \"Alice\", \"age\": 30, \"active\": true, \"nested\": {\"key\": null}}'. "
             "Print parsed['name'] == 'Alice'."),
            ["json_parser.py"],
            [
                {"type": "execution", "file": "json_parser.py", "expect": "Alice"},
                {"type": "execution", "file": "json_parser.py", "expect": "null"},
                {"type": "execution", "file": "json_parser.py", "expect": "nested"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T21", "Streaming Median Calculator",
            "Heaps and online algorithms",
            ("Create streaming_median.py that maintains median of a stream using two heaps (max-heap for lower half, "
             "min-heap for upper half). Implement add(num) and get_median(). Demonstrate with stream: [5, 15, 1, 3, 8, 7, 9, 10]. "
             "After each insertion, print current median. Final output must show medians: 5.0, 10.0, 5.0, 4.0, 5.0, 6.0, 7.0, 7.5."),
            ["streaming_median.py"],
            [
                {"type": "execution", "file": "streaming_median.py", "expect": "7.5"},
                {"type": "execution", "file": "streaming_median.py", "expect": "5.0"},
                {"type": "execution", "file": "streaming_median.py", "expect": "10.0"},
                {"type": "execution", "file": "streaming_median.py", "expect": "O(log n)"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T22", "Template Engine",
            "String interpolation and control flow",
            ("Create template_engine.py with a simple template renderer. Support: {{var}} variable substitution, "
             "{% for item in items %}...{% endfor %} loops, and {% if condition %}...{% endif %} conditionals. "
             "No external template libraries. Render: 'Hello {{name}}! You have {{count}} messages.' "
             "with name='Alice', count=5. Output must contain 'Hello Alice! You have 5 messages.'"),
            ["template_engine.py"],
            [
                {"type": "execution", "file": "template_engine.py",
                 "expect": "Hello Alice! You have 5 messages."},
                {"type": "execution", "file": "template_engine.py", "expect": "for"},
                {"type": "execution", "file": "template_engine.py", "expect": "if"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T23", "Token Bucket Rate Limiter",
            "Algorithm design and time-based state",
            ("Create rate_limiter.py implementing a token bucket algorithm. Class TokenBucket with capacity and refill_rate. "
             "allow_request(tokens_needed) returns True if tokens available, else False. Tokens refill over time. "
             "Demonstrate: bucket=capacity=10, rate=1/sec. Request 5 (True), request 7 (False), wait 2s, request 7 (True). "
             "Output must show 'Allowed: True/False' pattern matching expected behavior."),
            ["rate_limiter.py"],
            [
                {"type": "execution", "file": "rate_limiter.py", "expect": "Allowed"},
                {"type": "execution", "file": "rate_limiter.py", "expect": "False"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T24", "Sudoku Solver with Constraint Propagation",
            "Backtracking and constraint satisfaction",
            ("Create sudoku_solver.py that solves 9x9 Sudoku using backtracking + constraint propagation (forward checking). "
             "Input is a grid with 0 for empty cells. Print solved grid. Must solve hard puzzle in <2 seconds. "
             "Puzzle: [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]"),
            ["sudoku_solver.py"],
            [
                {"type": "execution", "file": "sudoku_solver.py", "expect": "Solved"},
                {"type": "execution", "file": "sudoku_solver.py", "expect": "8"},
                {"type": "execution", "file": "sudoku_solver.py", "expect": "2"}
            ],
            6, 120
        ))
        cases.append(TestCase(
            "T25", "Custom Event Loop",
            "Async primitives and callback scheduling",
            ("Create event_loop.py implementing a minimal async event loop from scratch. No asyncio module. "
             "Support: call_soon(callback), run_until_complete(), and basic sleep(delay) coroutine. "
             "Demonstrate scheduling 3 tasks that print 'Task N done' after staggered delays. "
             "Output must contain 'Task 1 done', 'Task 2 done', 'Task 3 done' in correct order."),
            ["event_loop.py"],
            [
                {"type": "execution", "file": "event_loop.py", "expect": "Task 3 done"},
                {"type": "execution", "file": "event_loop.py", "expect": "Task 1 done"},
                {"type": "execution", "file": "event_loop.py", "expect": "Task 2 done"}
            ],
            6, 120
        ))
        return cases

    def run(self, agent: CodingAgent, test_ids: Optional[List[str]] = None):
        results = []
        for case in self.cases:
            if test_ids and case.id not in test_ids:
                continue
            agent._abort = False
            agent.llm.reset_abort()
            agent.llm.reset_cooldowns()
            print(f"\n--- {case.id} {case.name} ---")
            start = time.time()
            out_holder = [None]
            exc_holder = [None]
            def _run_case():
                try:
                    out_holder[0] = agent.run(
                        case.task_prompt, case.max_turns,
                        case.expected_outputs, case.validation_checks
                    )
                except Exception as e:
                    exc_holder[0] = e    
            t = threading.Thread(target=_run_case, daemon=True)
            t.start()
            t.join(timeout=case.timeout)
            dur = time.time() - start
            if exc_holder[0]:
                print(f"✗ Exception: {exc_holder[0]}")
                traceback.print_exc()
                results.append((case.id, False, dur, str(exc_holder[0])))
                continue
            if t.is_alive():
                print(f"✗ Timed out after {case.timeout}s")
                agent._abort = True
                agent.llm.abort()
                results.append((case.id, False, dur, "TIMEOUT"))
                t.join(timeout=2.0)
                continue
            out = out_holder[0] or ""
            ok = "SUCCESS" in out
            if ok:
                print(f"✓ Passed in {dur:.1f}s")
            else:
                print(f"✗ Failed in {dur:.1f}s")
            results.append((case.id, ok, dur, out))
        return results

def run_single_task(agent: CodingAgent) -> None:
    task = input("Enter task description: ").strip()
    if not task:
        print("Empty task, cancelling.")
        return
    expected = input("Expected output files (comma-separated, optional): ").strip()
    expected_outputs = [f.strip() for f in expected.split(",") if f.strip()] if expected else []
    checks = []
    if expected_outputs:
        add_checks = input("Add validation checks? (y/n): ").strip().lower()
        if add_checks in ("y", "yes"):
            for fp in expected_outputs:
                if fp.endswith(".py") or fp.endswith(".cpp"):
                    expect_str = input(f"Expected output substring for {fp} (optional): ").strip()
                    stdin_str = input(f"Stdin input for {fp} (optional): ").strip()
                    check = {"type": "execution", "file": fp}
                    if expect_str:
                        check["expect"] = expect_str
                    if stdin_str:
                        check["input"] = stdin_str
                    checks.append(check)
                else:
                    checks.append({"type": "file_exists", "path": fp})
    max_turns = input("Max turns (default 6): ").strip()
    max_turns = int(max_turns) if max_turns.isdigit() else 6
    result = agent.run(task, max_turns=max_turns, expected_outputs=expected_outputs, validation_checks=checks)
    print(f"\nResult: {result}")

def run_test_suite(agent: CodingAgent, suite: TestSuite) -> None:
    test_ids = input("Test IDs to run (comma-separated, or 'all'): ").strip()
    if test_ids.lower() in ("", "all"):
        test_ids = None
    else:
        test_ids = [t.strip() for t in test_ids.split(",") if t.strip()]
    results = suite.run(agent, test_ids)
    print("\n--- Test Results ---")
    for tid, ok, dur, msg in results:
        print(f"  {tid}: {'PASS' if ok else 'FAIL'} ({dur:.1f}s) {msg[:100]}")

def main():
    config = AgentConfig()
    try:
        agent = CodingAgent(config)
    except RuntimeError as e:
        print(f"Initialization error: {e}")
        print("Make sure at least one API key is set in environment variables:")
        print("  GROQ_API_KEY, OPENROUTER_API_KEY, or GOOGLE_API_KEY")
        return
    suite = TestSuite()
    while True:
        print("\n" + "=" * 50)
        print("1. Single task")
        print("2. Run tests")
        print("3. Stats")
        print("4. Quit")
        choice = input("Choice: ").strip().lower()
        if choice in ("4", "quit", "exit"):
            print("Goodbye.")
            break
        elif choice == "1":
            run_single_task(agent)
        elif choice == "2":
            run_test_suite(agent, suite)
        elif choice == "3":
            agent.semantic_memory.print_stats()
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
    
