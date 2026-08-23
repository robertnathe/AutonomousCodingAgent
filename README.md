Advanced Coding Agent

An autonomous AI coding agent that plans, writes, executes, debugs, and iterates on Python and C++ programs using 
large language models. It combines multi-backend LLM routing, semantic memory of past solutions, failure-pattern 
learning, task decomposition, and a rich tool-using loop to solve programming tasks with minimal human intervention.

Features

- Multi-provider LLM support – Groq, OpenRouter, and Google Gemini with automatic fallback and rate limiting
- Semantic memory – TF-IDF + hybrid cosine/Jaccard retrieval of previous successful solutions and plans;
- learns from outcomes
- Failure memory – Stores error signatures and hints so the agent can avoid repeating the same mistakes
- Code execution sandbox – Runs Python and C++ (with auto-compilation, dependency detection, and limited auto-install)
- Planning & decomposition – Breaks complex tasks into sub-goals and generates executable action plans
- Tool-using agent loop – `write_file`, `execute_file`, `read_file`, shell commands, validation, and `finish`
- Reflection – Analyzes failures and suggests capability improvements
- Built-in test suite – 25 diverse coding challenges covering concurrency, algorithms, multi-language bridges,
- security, parsing, etc.
- Safety-oriented runner – Optional Docker sandbox script that limits CPU/memory and runs as a non-root user

Requirements

- Python 3.10+
- `requests`
- Optional: `g++` (for C++ compilation), Docker (for the sandboxed runner)

API keys (at least one required):

```bash
export GROQ_API_KEY="gsk_..."
export OPENROUTER_API_KEY="sk-or-..."
export GOOGLE_API_KEY="AIza..."
```

Install minimal dependencies:

```bash
pip install requests
```

Quick Start

```bash
# Clone / copy the repository
cd path/to/agent

# Make sure at least one API key is set
export GROQ_API_KEY="your-key"

# Run interactively
python AdvancedCodingAgent.py
```

You will see a simple menu:

```
1. Single task
2. Run tests
3. Stats
4. Quit
```

Single task

Enter a natural-language task description, optional expected output files, and optional validation checks. 
The agent will plan, write code, execute it, and iterate until success or the turn limit is reached.

Running the test suite

Choose option 2 and enter test IDs (e.g. `T1,T5,T16`) or `all`. Each test has a timeout and validation criteria 
(file existence + expected stdout substrings).

Sandboxed execution (recommended)

```bash
chmod +x run-agent-safely.sh
./run-agent-safely.sh
```

This builds a minimal Docker image, mounts the current directory, drops privileges to UID 1000, and 
limits the container to 2 CPUs / 3 GB RAM.

Configuration

Most settings live in the `AgentConfig` dataclass near the top of `AdvancedCodingAgent.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `models` | llama-3.3-70b / gemini-… | Model IDs per backend |
| `backend_priority` | `["groq", "openrouter", "google"]` | Fallback order |
| `global_rpm` | 30 | Global rate-limit target |
| `max_actions_per_turn` | 6 | Safety limit on tool calls |
| `max_execution_timeout` | 30 s | Per-process timeout |
| `max_memory_entries` | 50 | Semantic memory capacity |
| `semantic_similarity_threshold` | 0.72 | Retrieval threshold |
| `evolution.enabled` | `True` | Enable reflection & learning |

Memory is persisted under `./agent_memory_db/` (JSON files).

Architecture Overview

```
CodingAgent
├── LLMClient (Groq / OpenRouter / Google) + cache + rate limiter
├── SemanticMemoryManager
│   ├── Exact + inverted-index retrieval
│   ├── Hybrid cosine / Jaccard scoring with outcome feedback
│   └── FailurePattern store
├── CodeExecutor
│   ├── Python (with ModuleNotFoundError → pip install retry)
│   └── C++ (g++ -std=c++17, auto -pthread / -lcurl / etc.)
├── Planner / Decomposer / Reflection modules
└── ToolExecutor (write_file, execute_file, finish, …)
```

The main loop:

1. Try to replay a high-scoring cached plan.
2. Otherwise generate a fresh plan and execute it.
3. Enter the reactive turn loop: LLM → extract actions → execute → observe → reflect.
4. On success, store the solution (and optionally the plan) in semantic memory.
5. On failure, store the error signature and any useful hints.

Test Suite Highlights

| ID | Focus |
|----|-------|
| T1  | Debugging a segfault from logs |
| T2  | Concurrent SHA-256 hashing |
| T3  | C++ → JSON → Python multi-language bridge |
| T5  | Secure SQLite (parameterized queries + injection demo) |
| T16 | LRU cache with TTL |
| T18 | Mini regex engine (no `re` module) |
| T20 | JSON parser from scratch |
| T24 | Sudoku solver with constraint propagation |
| T25 | Custom async event loop |

See the `TestSuite` class for the full list and exact validation criteria.

License

This project is licensed under the GNU Affero General Public License v3.0.  
See the [LICENSE](LICENSE) file for the full text.

Because of the AGPL, if you run a modified version of this agent as a network service, you must make the 
corresponding source available to users of that service.

Safety Notes

- The agent can execute arbitrary code that it generates. Always run it inside the provided Docker sandbox
- (or an equivalent isolated environment) when experimenting with untrusted tasks.
- Auto-install of Python packages is enabled by default but limited to a small retry budget.
- Network access, file system, and process limits should be constrained in production deployments.

Contributing

Bug reports, additional test cases, and improvements to the planning / memory / reflection modules are welcome. 
Please keep the AGPL-3.0 license notice intact.
