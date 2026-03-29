import hashlib, json, os, re, shutil
import subprocess, sys, time, traceback
import fcntl, yaml, requests, random
import threading
import json as json_module
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
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
    max_cpp_compile_timeout: int = 60
    max_cpp_execution_timeout: int = 30
@dataclass
class EvolutionConfig:
    enabled: bool = True
    self_modify: bool = False  # Master switch for self-modification (DANGEROUS)
    auto_reflect: bool = True
    test_suite_path: str = "./agent_test_suite"
    max_context_turns: int = 6  # Rolling window for context compression
    capability_registry_file: str = "./capabilities.yaml"
    min_fitness_threshold: float = 0.8
@dataclass
class TestCase:
    id: str
    name: str
    description: str
    task_prompt: str
    expected_outputs: List[str]
    validation_checks: List[Dict[str, Any]]
    max_turns: int = 15
    timeout: int = 120

class ReflectionEngine:   
    def __init__(self, memory_manager: 'MemoryManager'):
        self.memory = memory_manager
        self.patterns: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "last_seen": 0})
        self.failure_log: List[Dict] = []       
    def analyze_session(self, task: str, success: bool, observations: str, turn_count: int) -> Dict:
        analysis = {
            "timestamp": time.time(),
            "task_hash": hashlib.md5(task.encode()).hexdigest()[:8],
            "success": success,
            "turn_count": turn_count,
            "error_type": None,
            "bottleneck": None,
            "suggested_capability": None
        }
        if not success:
            if "timeout" in observations.lower() or turn_count >= 9:
                analysis["error_type"] = "timeout"
                analysis["bottleneck"] = "inefficient_algorithm_or_infinite_loop"
                analysis["suggested_capability"] = "complexity_analyzer"
            elif "return code: 1" in observations.lower():
                analysis["error_type"] = "execution_error"
                if "sqlite" in task.lower():
                    analysis["suggested_capability"] = "sql_debugger"
                elif "import" in observations.lower():
                    analysis["suggested_capability"] = "dependency_resolver"
            elif "no such file" in observations.lower():
                analysis["error_type"] = "missing_file"
                analysis["suggested_capability"] = "path_validator"
            elif "outcome: continue" in observations.lower():
                analysis["error_type"] = "incomplete_task"
                analysis["suggested_capability"] = "task_decomposer"
            pattern_key = analysis["error_type"] or "unknown"
            self.patterns[pattern_key]["count"] += 1
            self.patterns[pattern_key]["last_seen"] = time.time()
            self.failure_log.append(analysis)
        return analysis
    def suggest_evolution(self) -> Optional[str]:
        suggestions = []
        for error_type, data in self.patterns.items():
            if data["count"] >= 2:
                if error_type == "timeout":
                    suggestions.append("Add automatic complexity analysis before code generation")
                elif error_type == "execution_error":
                    suggestions.append("Add pre-execution syntax validation")
                elif error_type == "incomplete_task":
                    suggestions.append("Improve task decomposition for multi-step problems")
        return suggestions[0] if suggestions else None
    def get_stats(self) -> Dict:
        return {
            "total_failures": len(self.failure_log),
            "pattern_counts": dict(self.patterns),
            "suggestion": self.suggest_evolution()
        }

class CapabilityRegistry:
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.tools: Dict[str, Dict] = {}
        self.load()
    def load(self):
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = yaml.safe_load(f) or {}
                    self.tools = data.get("tools", {})
            except Exception as e:
                print(f"[Registry] Load error: {e}")
    def save(self):
        try:
            with open(self.registry_path, "w") as f:
                yaml.dump({"tools": self.tools, "version": time.time()}, f)
        except Exception as e:
            print(f"[Registry] Save error: {e}")
    def register_from_session(self, task: str, code: str, description: str) -> bool:
        func_match = re.search(r'def\s+(\w+)\s*\([^)]*\):', code)
        if func_match:
            func_name = func_match.group(1)
            if func_name not in self.tools:
                self.tools[func_name] = {
                    "description": description,
                    "code": code,
                    "source_task": task[:100],
                    "created": time.time(),
                    "uses": 0
                }
                self.save()
                return True
        return False
    def get_tool(self, name: str) -> Optional[Dict]:
        tool = self.tools.get(name)
        if tool:
            tool["uses"] += 1
            self.save()
        return tool

class TestSuite:
    def __init__(self, suite_path: str):
        self.suite_path = Path(suite_path)
        self.suite_path.mkdir(exist_ok=True)
        self.results: List[Dict] = []
        self.cases = self._define_default_cases()
    def _define_default_cases(self) -> List[TestCase]:
        return [
            TestCase(
                id="T1",
                name="Prime Factor Optimization",
                description="Find largest prime factor with optimization on timeout",
                task_prompt="""Create a script prime_factor.py that finds the largest prime factor of 600851475143. 
                Start with basic trial division, but if execution exceeds 1 second, refactor to O(√n) algorithm. 
                Print result and time taken.""",
                expected_outputs=["prime_factor.py"],
                validation_checks=[{"type": "file_exists", "path": "prime_factor.py"}],
                max_turns=4
            ),
            TestCase(
                id="T2",
                name="Import Restructuring",
                description="Handle directory moves and import updates",
                task_prompt="""1. Create math_lib/operations.py with power(a,b) function. Create app.py importing it and printing 2**10.
                2. Move operations.py to core/utils/, update imports, verify output still 1024.""",
                expected_outputs=["app.py", "core/utils/operations.py"],
                validation_checks=[{"type": "execution", "file": "app.py", "expect": "1024"}],
                max_turns=4
            ),
            TestCase(
                id="T3",
                name="Vending Machine OOP",
                description="Custom exceptions and state management",
                task_prompt="""Create vending.py with VendingMachine class and InsufficientFundsError. 
                Demonstrate: deposit $2.00, try buying $2.50 item (catch error), deposit $1 more, buy successfully.
                Print balance and inventory.""",
                expected_outputs=["vending.py"],
                validation_checks=[{"type": "execution", "file": "vending.py"}],
                max_turns=4
            ),
            TestCase(
                id="T4",
                name="Data Cleaning Pipeline",
                description="Handle dirty CSV data with type validation",
                task_prompt="""Create raw_data.csv (Name, Age, Salary) with 6 rows including non-numeric ages/empty salaries.
                Create cleaner.py to filter invalid rows, calculate average salary, save to processed_data.json.""",
                expected_outputs=["raw_data.csv", "cleaner.py", "processed_data.json"],
                validation_checks=[{"type": "file_exists", "path": "processed_data.json"}],
                max_turns=4
            ),
            TestCase(
                id="T5",
                name="Pi Approximation",
                description="Numerical methods and convergence",
                task_prompt="""Create pi_approx.py using Leibniz formula: π = 4 * Σ((-1)^n/(2n+1)).
                Iterate until |approximation - math.pi| < 10^-5. Print iterations needed.""",
                expected_outputs=["pi_approx.py"],
                validation_checks=[{"type": "execution", "file": "pi_approx.py"}],
                max_turns=4
            ),
            TestCase(
                id="T6",
                name="Grid BFS Pathfinding",
                description="Graph traversal and obstacle handling",
                task_prompt="""Create grid_bfs.py with 10x10 grid, obstacles. Implement BFS from (0,0) to (9,9).
                Output path coordinates and step count. Handle no-path case.""",
                expected_outputs=["grid_bfs.py"],
                validation_checks=[{"type": "execution", "file": "grid_bfs.py"}],
                max_turns=4
            ),
            TestCase(
                id="T7",
                name="SQLite Relational Query",
                description="Complex JOINs and subqueries",
                task_prompt="""Create SQLite database with Departments and Employees tables.
                Insert 3 departments, 10 employees. Query: find employees earning more than their dept average.
                Save to high_earners.json. Use proper foreign keys.""",
                expected_outputs=["high_earners.json"],
                validation_checks=[{"type": "file_exists", "path": "high_earners.json"}],
                max_turns=4
            ),
            TestCase(
                id="T8",
                name="Concurrent Hashing",
                description="Parallel processing with concurrent.futures",
                task_prompt="""Create concurrent_hash.py scanning for .txt/.py files, calculate SHA-256 in parallel.
                Store {filename: hash} dict. Print execution time and verify count matches.""",
                expected_outputs=["concurrent_hash.py"],
                validation_checks=[{"type": "execution", "file": "concurrent_hash.py"}],
                max_turns=4
            ),
            TestCase(
                id="T9",
                name="Memoization Decorator",
                description="Higher-order functions and performance",
                task_prompt="""Create memoize decorator and apply to recursive Fibonacci.
                Calculate F(50) with timing. Compare to F(30) without decorator.""",
                expected_outputs=["memoize_fib.py"],
                validation_checks=[{"type": "execution", "file": "memoize_fib.py"}],
                max_turns=4
            ),
            TestCase(
                id="T10",
                name="Monte Carlo Integration",
                description="Probabilistic numerical methods",
                task_prompt="""Create monte_carlo.cpp to estimate ∫sin(x)dx from 0 to π using 1,000,000 random points.
                Compare estimate to analytical value 2. Print absolute error.""",
                expected_outputs=["monte_carlo.cpp"],
                validation_checks=[{"type": "execution", "file": "monte_carlo.cpp"}],
                max_turns = 4
            )
        ]
    def validate_result(self, case: TestCase, agent: 'CodingAgent') -> Tuple[bool, List[str]]:
        passed = True
        details = []
        for check in case.validation_checks:
            if check["type"] == "file_exists":
                if not Path(check["path"]).exists():
                    passed = False
                    details.append(f"Missing file: {check['path']}")
                else:
                    details.append(f"✓ Found: {check['path']}")
            elif check["type"] == "execution":
                file_path = check["file"]
                if Path(file_path).exists():
                    result = agent.execute_file(file_path)
                    if "Return code: 0" not in result:
                        passed = False
                        details.append(f"Execution failed: {result[:200]}")
                    else:
                        details.append(f"✓ Executed: {file_path}")
                        if "expect" in check:
                            if check["expect"] not in result:
                                passed = False
                                details.append(f"Expected output '{check['expect']}' not found")
                else:
                    passed = False
                    details.append(f"File not found for execution: {file_path}")
        return passed, details
    def run_evaluation(self, agent: 'CodingAgent', specific_tests: Optional[List[str]] = None) -> Dict:
        print(f"\n{'='*60}")
        print("RUNNING TEST SUITE EVALUATION")
        print(f"{'='*60}")
        results = []
        cases_to_run = [c for c in self.cases if not specific_tests or c.id in specific_tests]
        for case in cases_to_run:
            print(f"\n[{case.id}] {case.name}")
            print("-" * 40)
            for f in case.expected_outputs:
                if Path(f).exists():
                    Path(f).unlink()
            start_time = time.time()
            try:
                output = agent.run(case.task_prompt, max_turns=case.max_turns)
                duration = time.time() - start_time
                success = "outcome: success" in output.lower() or "✓ Task completed" in output
                validated, details = self.validate_result(case, agent)
                final_pass = success and validated
                result = {
                    "id": case.id,
                    "name": case.name,
                    "success": final_pass,
                    "llm_claimed_success": success,
                    "validation_passed": validated,
                    "duration": duration,
                    "details": details,
                    "output_snippet": output[-500:]
                }
            except Exception as e:
                result = {
                    "id": case.id,
                    "name": case.name,
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            results.append(result)
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(f"{status} in {result.get('duration', 0):.1f}s")
            for d in result.get('details', []):
                print(f"  {d}")
        passed = sum(1 for r in results if r["success"])
        total = len(results)
        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed}/{total} passed ({passed/total*100:.0f}%)")
        print(f"{'='*60}")
        report_file = self.suite_path / f"report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump({"results": results, "summary": {"passed": passed, "total": total}}, f, indent=2)
        print(f"Report saved to: {report_file}")
        return {"passed": passed, "total": total, "results": results}

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
    def __init__(self, backup_dir: str, executor: Optional['CodeExecutor'] = None):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.last_written_file: Optional[str] = None
        self._executor = executor
        self._version_index: Dict[str, int] = {}
        self._content_index: Dict[str, str] = {}
        self._wal_path = self.backup_dir / ".wal_index"
        self._lock = threading.RLock()
        self._recover_index()
    def _compute_hash(self, content: str) -> str:
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()
    def _recover_index(self):
        if self._wal_path.exists():
            try:
                with open(self._wal_path, 'r') as f:
                    data = json.load(f)
                    self._version_index = data.get('versions', {})
                    self._content_index = data.get('contents', {})
            except (json.JSONDecodeError, IOError):
                pass
    def _persist_index(self):
        temp_path = self._wal_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump({
                'versions': self._version_index,
                'contents': self._content_index
            }, f)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(self._wal_path)
    def backup_file(self, file_path: str) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            return None
        with self._lock:
            try:
                content = path.read_text(encoding="utf-8")
                content_hash = self._compute_hash(content)
                if content_hash in self._content_index:
                    existing_path = Path(self._content_index[content_hash])
                    if existing_path.exists():
                        next_version = self._version_index.get(file_path, 0) + 1
                        backup_path = self.backup_dir / f"{path.name}_v{next_version}.bak"
                        try:
                            existing_path.link_to(backup_path)  # Hard link
                            self._version_index[file_path] = next_version
                            self._persist_index()
                            return str(backup_path)
                        except (OSError, AttributeError):
                            pass
                next_version = self._version_index.get(file_path, 0) + 1
                backup_path = self.backup_dir / f"{path.name}_v{next_version}.bak"
                shutil.copy2(path, backup_path)
                self._version_index[file_path] = next_version
                self._content_index[content_hash] = str(backup_path)
                self._persist_index()
                return str(backup_path)
            except Exception as e:
                print(f"Warning: Failed to backup {file_path}: {e}")
                return None
    def write_file(self, file_path: str, content: str) -> str:
        if not file_path or not isinstance(file_path, str) or not file_path.strip():
            return "CRITICAL ERROR: invalid file_path!"
        if file_path.lower().endswith(('.py', '.python')) and self._executor:
            try:
                syntax_check = self._executor.validate_python_syntax(content)
                if "FAILED" in syntax_check:
                    return (
                        f"{syntax_check}\n"
                        f"(Writing blocked to prevent creation of invalid Python file: {file_path})"
                    )
            except Exception:
                pass
        if file_path.lower().endswith(('.cpp', '.cc', '.cxx', '.c++')) and self._executor:
            try:
                syntax_check = self._executor.validate_cpp_syntax(content)
                if "FAILED" in syntax_check:
                    return (
                        f"{syntax_check}\n"
                        f"(Writing blocked to prevent creation of invalid C++ file: {file_path})"
                    )
            except Exception:
                pass
        try:
            path = Path(file_path)
            self.backup_file(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix('.tmp')
            temp_path.write_text(content, encoding="utf-8")
            temp_path.replace(path)
            if path.suffix.lower() in {'.py', '.python', '.cpp', '.cc', '.cxx', '.c++'}:
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
        with self._lock:
            current_version = self._version_index.get(file_path, 0)
            if current_version <= 0:
                return "No backup found."
            target_version = current_version
            backup_path = self.backup_dir / f"{Path(file_path).name}_v{target_version}.bak"
            if not backup_path.exists():
                return f"Backup v{target_version} not found."
            try:
                path = Path(file_path)
                shutil.copy2(backup_path, path)
                self._version_index[file_path] = target_version - 1
                self._persist_index()
                return f"Reverted {file_path} to version {target_version}."
            except Exception as e:
                return f"Error reverting file: {e}"
    def get_deduplication_stats(self) -> Dict[str, any]:
        total_backups = sum(self._version_index.values())
        unique_contents = len(self._content_index)
        return {
            "total_backup_versions": total_backups,
            "unique_content_hashes": unique_contents,
            "deduplication_ratio": (total_backups - unique_contents) / max(total_backups, 1),
            "storage_saved_estimated": "O(n*m) space reduction for duplicate files"
        }

class CodeExecutor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.log_path = Path("./execution_traceback.log")
    def _get_cpp_compiler(self) -> Optional[str]:
        for compiler in ['g++', 'clang++', 'c++']:
            if shutil.which(compiler):
                return compiler
        return None
    def _execute_cpp_file(self, file_path: str) -> str:
        compiler = self._get_cpp_compiler()
        if not compiler:
            return "C++ compilation failed: No C++ compiler found (g++, clang++, or c++ required)"
        path = Path(file_path)
        if sys.platform == 'win32':
            exe_path = path.with_suffix('.exe')
        else:
            exe_path = path.parent / path.stem
        try:
            self.log_path.write_text("")
            compile_cmd = [
                compiler,
                '-std=c++20',
                '-O2',
                '-Wall',
                '-I/usr/include',
                '-I/usr/local/include',
                '-o', str(exe_path),
                str(file_path),
                '-lcurl',
                '-lyaml-cpp',
                '-lssl',
                '-lcrypto'
            ]
            compile_proc = subprocess.Popen(
                compile_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )
            try:
                compile_stdout, compile_stderr = compile_proc.communicate(timeout=self.config.max_cpp_compile_timeout)
                compile_rc = compile_proc.returncode
            except subprocess.TimeoutExpired:
                compile_proc.kill()
                compile_stdout, compile_stderr = compile_proc.communicate()
                compile_rc = -9
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"=== Compilation: {file_path} (rc={compile_rc}) at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Command: {' '.join(compile_cmd)}\n")
                f.write(compile_stderr or "(no stderr)\n")
                f.write("=" * 60 + "\n\n")
            if compile_rc != 0:
                return (
                    f"C++ Compilation FAILED (Return code: {compile_rc})\n"
                    f"Compiler: {compiler}\n"
                    f"STDOUT:\n{compile_stdout.strip() or '(none)'}\n"
                    f"STDERR:\n{compile_stderr.strip() or '(none)'}"
                )
            import stat
            if exe_path.exists():
                exe_path.chmod(exe_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            exe_to_run = f"./{exe_path.name}" if exe_path.parent == Path('.') else str(exe_path)
            run_proc = subprocess.Popen(
                [exe_to_run],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
            )
            try:
                run_stdout, run_stderr = run_proc.communicate(timeout=self.config.max_cpp_execution_timeout)
                run_rc = run_proc.returncode
            except subprocess.TimeoutExpired:
                run_proc.kill()
                run_stdout, run_stderr = run_proc.communicate()
                run_rc = -9
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"=== Execution: {exe_path} (rc={run_rc}) at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(run_stderr or "(no stderr)\n")
                f.write("=" * 60 + "\n\n")
            try:
                if exe_path.exists():
                    exe_path.unlink()
            except Exception:
                pass
            return (
                f"C++ Compilation: SUCCESS\n"
                f"Execution Return code: {run_rc}\n"
                f"STDOUT:\n{run_stdout.strip() or '(none)'}\n"
                f"STDERR:\n{run_stderr.strip() or '(none)'}"
            )
        except Exception as e:
            return f"C++ execution failed: {e}"
    def execute_file(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {'.py', '.python'}:
            result = self._execute_python_file(file_path)  # ✅ Fixed
        elif suffix in {'.cpp', '.cc', '.cxx', '.c++'}:
            result = self._execute_cpp_file(file_path)      # ✅ Fixed
        else:
            result = f"Unsupported file type: {suffix}"
        return result
    def _execute_python_file(self, file_path: str) -> str:
        try:
            self.log_path.write_text("")
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
                f"STDERR:\n{stderr.strip() or '(none)'}"
            )
        except Exception as e:
            return f"Execution failed: {e}"
    def execute_code(self, code: str, language: str = "python") -> str:
        if language.lower() == "python":
            syntax_check = self.validate_python_syntax(code)
            if "FAILED" in syntax_check:
                return syntax_check + "\n(Execution skipped – fix syntax first)"
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
                return f"Return code: {rc}\nSTDOUT:\n{stdout.strip() or '(none)'}\nSTDERR:\n{stderr.strip() or '(none)'}"
            except Exception as e:
                return f"Execution failed: {e}"
        else:
            return f"Direct code execution not supported for {language}. Use write_file then execute_file."
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
    def validate_python_syntax(self, code: str) -> str:
        if not isinstance(code, str):
            return "Invalid code type for validation"
        try:
            compile(code, "<agent_generated_code>", "exec")
            return "Python syntax validation PASSED."
        except SyntaxError as e:
            return f"Python syntax validation FAILED: {e.msg} (line {e.lineno or 'unknown'})"
        except Exception as e:
            return f"Validation error: {str(e)[:100]}"
    def validate_cpp_syntax(self, code: str) -> str:
        if not isinstance(code, str):
            return "Invalid code type for validation"
        errors = []
        lines = code.split('\n')
        open_braces = code.count('{') - code.count('}')
        if open_braces != 0:
            errors.append(f"Unbalanced braces: {open_braces} extra {'{' if open_braces > 0 else '}'})")
        open_parens = code.count('(') - code.count(')')
        if open_parens != 0:
            errors.append(f"Unbalanced parentheses: {open_parens} extra '(' if open_parens > 0 else ')'")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (stripped and 
                not stripped.startswith('#') and 
                not stripped.startswith('//') and 
                not stripped.startswith('/*') and
                not stripped.startswith('*') and
                not stripped in ['{', '}', ''] and
                not stripped.endswith('{') and
                not stripped.endswith('}') and
                not stripped.endswith(')') and
                not stripped.endswith('//') and
                not any(stripped.startswith(kw) for kw in ['if', 'for', 'while', 'switch', 'else', 'do', 'class', 'struct', 'namespace', 'public:', 'private:', 'protected:', '#'])):
                if not stripped.endswith(';') and not stripped.endswith(':'):
                    errors.append(f"Line {i}: Possible missing semicolon: {stripped[:50]}")
        if 'int main(' not in code and 'int main (' not in code:
            errors.append("Warning: No 'int main()' function found")
        if errors:
            return f"C++ syntax validation FAILED:\n" + "\n".join(errors)
        return "C++ syntax validation PASSED (basic checks)."
    def optimize_code(self, code: str, language: str = "python") -> str:
        if language.lower() != "python":
            return f"Optimization not supported for {language}"
        try:
            import ast
            import tempfile
            tree = ast.parse(code)
            optimized_code = compile(tree, '<optimized>', 'exec')
            if shutil.which('pylint'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.flush()
                    result = subprocess.run(['pylint', '--output-format=json', f.name], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        issues = json.loads(result.stdout)
            if shutil.which('autopep8'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.flush()
                    result = subprocess.run(['autopep8', '--in-place', f.name], timeout=10)
                    if result.returncode == 0:
                        with open(f.name, 'r') as f:
                            optimized_code = f.read()
            return optimized_code
        except Exception as e:
            return f"Optimization failed: {e}. Using original code.\n{code}"

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
    IGNORED_DIRS = {".git", "venv", "__pycache__", "node_modules", ".venv", ".agent_backups"}
    def __init__(self, config: Optional[AgentConfig] = None, evolution_config: Optional[EvolutionConfig] = None):
        self.config = config or AgentConfig()
        self.evolution_config = evolution_config or EvolutionConfig()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=api_key)
        self.executor = CodeExecutor(self.config)
        self.file_manager = FileManager(self.config.backup_dir, executor=self.executor)
        self.memory = MemoryManager(self.config)
        self.cache = ToolCache()
        self.reflection = ReflectionEngine(self.memory)
        self.capability_registry = CapabilityRegistry(self.evolution_config.capability_registry_file)
        self.execution_history: List[Dict] = []
        self.last_observations: str = ""
        print(f"[Agent] Initialized. Evolution: {'ON' if self.evolution_config.enabled else 'OFF'}")
        if self.evolution_config.enabled:
            print(f" - Auto-reflect: {self.evolution_config.auto_reflect}")
            print(f" - Context window: {self.evolution_config.max_context_turns} turns")
            print(f" - Self-modify: {'ENABLED (DANGER)' if self.evolution_config.self_modify else 'DISABLED'}")
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
    def execute_file(self, file_path: str) -> str:
        result = self.executor.execute_file(file_path)
        self.cache.clear()
        return result
    def execute_python_file(self, file_path: str) -> str:
        return self.execute_file(file_path)
    def execute_cpp_file(self, file_path: str) -> str:
        return self.execute_file(file_path)
    def run_python_code(self, code: str) -> str:
        return self.executor.execute_code(code, language="python")
    def run_cpp_code(self, code: str) -> str:
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                temp_file = f.name
            result = self.executor.execute_file(temp_file)
            try:
                os.unlink(temp_file)
            except Exception:
                pass
            return result
        except Exception as e:
            return f"C++ code execution failed: {e}"
    def install_package(self, package: str) -> str:
        return self.executor.install_package(package)
    def search_in_files(self, pattern: str, include_extensions: Optional[List[str]] = None) -> str:
        cache_key = self.cache.get_key("search_in_files", {"pattern": pattern, "include_extensions": include_extensions or []})
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
            pip_out = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, timeout=10).stdout.strip()
            site_out = subprocess.run([sys.executable, "-m", "site"], capture_output=True, text=True, timeout=10).stdout.strip()
            compiler = self.executor._get_cpp_compiler()
            compiler_info = f"C++ Compiler: {compiler}" if compiler else "C++ Compiler: NOT FOUND (install g++ or clang++)"
            result = f"pip list:\n{pip_out or '(empty)'}\n\npython -m site output:\n{site_out or '(empty)'}\n\n{compiler_info}"
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
            "ABSOLUTE RULES - VIOLATING THESE CAUSES IMMEDIATE FAILURE:\n"
            "- You MUST check the return code of execute_file (or execute_python_file/execute_cpp_file)\n"
            "- If return code != 0 → you MUST output outcome: continue\n"
            "- You MUST read the target output file and confirm it exists AND contains correct-looking data BEFORE declaring success\n"
            "- Never lie about success when observations show clear failure\n\n"
            "You have these tools:\n"
            "- list_directory_files()\n"
            "- read_file(file_path: str)\n"
            "- write_file(file_path: str, content: str)\n"
            "- make_directory(dir_path: str)\n"
            "- execute_file(file_path: str) - executes Python or C++ files\n"
            "- execute_python_file(file_path: str) - legacy, same as execute_file for .py\n"
            "- execute_cpp_file(file_path: str) - legacy, same as execute_file for .cpp\n"
            "- undo_last_change(file_path: str)\n"
            "- run_python_code(code: str)\n"
            "- run_cpp_code(code: str) - write, compile and execute C++ code\n"
            "- install_package(package: str)\n"
            "- search_in_files(pattern: str, include_extensions: list[str] = None)\n"
            "- get_environment_info()\n"
            "- clear_cache()\n\n"
            "FILE NAME & OUTPUT METHOD RULES – MUST BE FOLLOWED LITERALLY:"
            " When the task explicitly names a file to create ('create app.py', 'create memoize_fib.py', 'save to high_earners.json', etc.) → use exactly that filename with write_file. NEVER choose a different name like 'fibonacci.py' or 'database.py'."
            " When the task says 'print', 'printing', 'print the', 'display', 'show' and does NOT mention saving/writing/creating a file → use print() to stdout ONLY. Do NOT write any file and do NOT call read_file unless the task explicitly requires a saved file."
            " When the task requires both printing and saving → print to stdout AND write the same content to the named file."
            " This rule is critical — violating it causes test suite failures (T2, T9)."
            "STRICT OUTPUT FORMAT — FOLLOW EXACTLY:\n"
            "1. Step-by-step reasoning in plain text.\n"
            "2. **One or more** ```actions``` YAML blocks (use multiple for complex multi-step plans).\n"
            "3. **Exactly one** ```yaml``` outcome block at the very end.\n\n"
            "CRITICAL RULES FOR DATA TASKS (never ignore):\n"
            "• For cleaning dirty CSV/JSON data, ALWAYS use try/except around int() or float() conversion.\n"
            "• Never use only .isdigit() or truthy checks — they fail on decimals, negatives, 'abc', etc.\n"
            "• After writing cleaner.py, ALWAYS run it with execute_file, then use read_file on the output JSON to verify.\n"
            "• Only declare outcome: success after you have seen correct numeric results.\n\n"
            "CRITICAL RULES FOR C++ TASKS:\n"
            "• For C++ files (.cpp, .cc, .cxx), use write_file then execute_file (or execute_cpp_file).\n"
            "• The C++ compiler will be auto-detected (g++, clang++, or c++).\n"
            "• Check compilation output for errors before declaring success.\n"
            "• The executable will be automatically cleaned up after execution.\n\n"
            "Actions format example (no colon after ```actions):\n"
            "```actions\n"
            "- tool: write_file\n"
            "  args:\n"
            "    file_path: demo.cpp\n"
            "    content: |\n"
            "      #include <iostream>\n"
            "      int main() { std::cout << \"Hello\" << std::endl; return 0; }\n"
            "```\n\n"
            "Outcome block (must be last):\n"
            "```yaml\n"
            "outcome: success\n"
            "```\n"
            "or\n"
            "```yaml\n"
            "outcome: continue\n"
            "```\n"
            "CRITICAL RULES FOR DATABASE / FILE OUTPUT TASKS:\n"
            "• Never declare outcome: success until:\n"
            "  1. execute_file returned code 0\n"
            "  2. You ran read_file on the target output file\n"
            "  3. The file exists and contains plausible data matching the task\n"
            "• If execution fails → declare outcome: continue and propose fixes"
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
        declared = bool(re.search(r"```yaml\s*outcome:\s*success\s*```", text, re.IGNORECASE))
        if not declared:
            return False
        has_verification_error = False
        if hasattr(self, 'last_observations') and self.last_observations:
            obs_lower = self.last_observations.lower()
            error_indicators = [
                "no such file or directory",
                "error reading file",
                "output.txt not found",
                "failed to",
                "exception",
                "compilation failed"
            ]
            has_verification_error = any(ind in obs_lower for ind in error_indicators)
        if has_verification_error:
            print("[Guard] Blocking premature success: verification showed error or missing file")
            return False
        return True
    def _auto_verify_execution(self, observations: str) -> str:
        if "Tool execute_file Result:" in observations or "Tool execute_python_file Result:" in observations or "Tool execute_cpp_file Result:" in observations:
            return observations
        extra = ""
        last_file = self.file_manager.last_written_file
        if last_file:
            print(f"Auto-verification: Re-running last written file → {last_file}")
            exec_result = self.execute_file(last_file)
            extra += f"\n--- Auto Execution of {last_file} ---\n{exec_result}"
        common_output = "output.txt"
        if Path(common_output).exists():
            read_result = self.read_file(common_output)
            extra += f"\n--- Auto-read {common_output} ---\n{read_result}"
        elif "output.txt" in observations.lower():
            extra += "\n--- Warning: output.txt was requested but does not exist ---"
        return observations + extra if extra else observations
    def _prune_context(self, context: List[str]) -> List[str]:
        max_turns = self.evolution_config.max_context_turns
        if len(context) <= max_turns:
            return context
        return [context[0]] + context[-(max_turns-1):]
    def _reflect(self, task: str, success: bool, observations: str, turn_count: int):
        if not self.evolution_config.auto_reflect:
            return None
        analysis = self.reflection.analyze_session(task, success, observations, turn_count)
        if not success:
            print(f"[Reflection] Failure analysis: {analysis.get('error_type', 'unknown')}")
            print(f"[Reflection] Suggestion: {analysis.get('suggested_capability', 'none')}")
        return analysis
    def _speculative_execute(self, task: str, context: List[str]) -> Tuple[str, str]:
        branches_prompt = self._get_system_prompt() + "\nProvide 3 solution variants..."
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": "You are an expert coder."},
                {"role": "user", "content": "Provide 3 solution variants."}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        branches = json.loads(response.choices[0].message.content)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._test_branch, branch): branch_id 
                for branch_id, branch in branches.items()
            }
            results = {}
            for future in concurrent.futures.as_completed(futures):
                branch_id = futures[future]
                try:
                    success, metrics = future.result(timeout=self.config.max_execution_timeout)
                    results[branch_id] = (success, metrics)
                    result_hash = hashlib.blake2b(str(metrics).encode()).hexdigest()
                    if hasattr(self, 'known_failures') and result_hash in self.known_failures:
                        continue
                except Exception as e:
                    results[branch_id] = (False, str(e))
        best_branch = self._select_optimal_branch(results, task)
        return self._promote_branch(best_branch)
    def run(self, task_description: str, max_turns: int = 10) -> str:
        print(f"Starting task: {task_description[:100]}...\n")
        memory = self.memory.retrieve(task_description)
        session_context = []
        system_prompt = self._get_system_prompt()
        final_success = False
        final_output = ""
        for turn in range(max_turns):
            print(f"\n{'='*60}")
            print(f"Turn {turn + 1}/{max_turns}")
            print(f"{'='*60}\n")
            session_context = self._prune_context(session_context)
            if len(session_context) == self.evolution_config.max_context_turns:
                print(f"[Evolution] Context compressed to last {self.evolution_config.max_context_turns} turns")
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
                    max_tokens=2048,
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
            self.last_observations = observations
            print("--- Observations ---")
            print(observations)
            print("-" * 60 + "\n")
            success_declared = self._check_outcome(turn_text)
            if success_declared:
                observations = self._auto_verify_execution(observations)
                print("Auto-verified observations:\n", observations)
            analysis = self._reflect(task_description, success_declared, observations, turn + 1)
            guidance_line = ""
            if analysis and not success_declared:
                guidance = analysis.get("suggested_capability") or analysis.get("bottleneck")
                if guidance:
                    guidance_line = f"\nREFLECTION_GUIDANCE: Detected {analysis.get('error_type', 'unknown')} - Suggestion: {guidance}. Adjust next actions accordingly."
            session_context.append(
                f"Turn {turn+1} Output:\n{turn_text}\n\n"
                f"Observation:\n{observations}{guidance_line}"
            )
            final_output = turn_text
            if success_declared:
                print("✓ Task completed successfully.")
                self.memory.store(task_description, turn_text)
                final_success = True
                break
        if not final_success:
            print("✗ Task failed to complete within turn limit.")
            self.memory.store(task_description, session_context[-1] if session_context else "No output")
        return final_output

def main():
    agent_config = AgentConfig(
        max_execution_timeout=45,
        max_code_timeout=20,
        max_cpp_compile_timeout=60,
        max_cpp_execution_timeout=30
    )
    evolution_config = EvolutionConfig(
        enabled=True,
        auto_reflect=True,
        max_context_turns=6,
        self_modify=False  # Keep False until sandbox is implemented
    )
    agent = CodingAgent(agent_config, evolution_config)
    test_suite = TestSuite("./agent_test_suite")
    print("\nSelect mode:")
    print("1. Run single task")
    print("2. Run full test suite (T1-T10)")
    print("3. Run specific test (T10 - Monte Carlo C++)")
    print("4. Reflection stats")
    choice = input("\nChoice (1-4): ").strip()
    if choice == "2":
        results = test_suite.run_evaluation(agent)
        print(f"\nFinal Score: {results['passed']}/{results['total']}")
    elif choice == "3":
        print("\nRunning T10 (Monte Carlo integration in C++)...")
        results = test_suite.run_evaluation(agent, specific_tests=["T10"])
    elif choice == "4":
        stats = agent.reflection.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        print("Enter task description (press Enter twice when finished):")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        task = " ".join(lines).strip()
        if not task:
            task = """Task 11. Monte Carlo Integration
            Create a c++ program monte_carlo.cpp to estimate the area under the curve f(x) = sin(x) between x = 0 and x = π using a 
            Monte Carlo simulation. The program should generate 1,000,000 random points within a bounding box [0, π] × [0, 1] and 
            determine the fraction of points that fall below the curve. Compare the estimated result to the analytical value of 2 
            and print the absolute error."""
        max_turns = input("Max turns (default 4): ").strip()
        max_turns = int(max_turns) if max_turns.isdigit() else 4
        result = agent.run(task, max_turns=max_turns)
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(result)

if __name__ == "__main__":
    main()
