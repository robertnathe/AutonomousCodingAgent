
# Autonomous Coding Agent

This project provides an autonomous coding agent that can read and write files, execute Python code, install packages, and iteratively refine solutions to coding tasks using the Groq API and optional persistent vector memory via ChromaDB.

## Features

- Runs an LLM-powered coding loop using Groq’s `chat.completions` API with a configurable model (default: `llama-3.3-70b-versatile`).  
- Persistent, versioned file backups for safe editing and undo (`.agent_backups`).
- Optional long-term memory using ChromaDB (`agent_memory_db`), storing task/result snippets for future retrieval. 
- Tooling layer for:
  - Listing files in the current project tree (with common directories ignored).  
  - Reading and writing files with automatic directory creation.
  - Executing Python files and inline Python code with timeouts and stderr logging (`execution_traceback.log`).
  - Installing Python packages via `pip` with timeouts.
  - Grep-based search across the repository with ignore rules. 
  - Inspecting the Python environment (`pip list`, `python -m site`).
- Simple in-memory cache to avoid redundant tool calls within a run.
- Enforced agent output protocol using YAML-based ```actions``` blocks to describe tool calls and a final ```yaml``` outcome block declaring success or continuation.

## Architecture

The core components are:

- `AgentConfig`: Dataclass holding model ID, backup directory, memory DB path, collection name, and timeout settings for execution and installs.
- `MemoryManager`: Wraps ChromaDB (if installed) to store and retrieve prior tasks and results; gracefully degrades if Chroma is unavailable.
- `FileManager`: Handles file backup, read/write, and undo of the last change via versioned `.bak` files in `.agent_backups`.
- `CodeExecutor`: Runs Python files or inline code in subprocesses with timeouts, capturing output and writing stderr to `execution_traceback.log`. 
- `ToolCache`: Simple dictionary-based cache keyed by a hash of tool name and arguments.  
- `CodingAgent`: High-level agent that:
  - Exposes tools (list/read/write files, execute code, install packages, search, etc.).  
  - Constructs a system prompt describing the available tools and strict output format. 
  - Streams responses from Groq, parses ```actions``` YAML blocks, executes tools, and feeds observations back into the session context.
  - Manages multi-turn task execution, long-term memory retrieval/storage, and optional auto-verification by rerunning the last written Python file when the model declares success.

## Requirements

- Python 3.9+ (recommended).  
- A valid Groq API key exported as `GROQ_API_KEY` in your environment; the agent will raise an error if it is missing.
- Python packages:
  - `groq`  
  - `pyyaml` (`yaml`)  
  - Optional: `chromadb` (for persistent vector memory)  

You can install the core dependencies with:

```bash
pip install groq pyyaml chromadb
```

If `chromadb` is not installed, memory support is disabled but the agent still runs.

## Usage

1. Set your Groq API key:

   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

2. Run the agent script:

   ```bash
   python autonomous_coding_agent.py
   ```

By default, `main()` constructs a task asking the agent to create `pi_approx.py` that approximates $\pi$ using the Leibniz series until the absolute error is below $10^{-5}$, then prints the number of iterations required.
The agent will interact with the filesystem and tools over several turns (up to `max_turns`, default 3) to complete this task.

### Integrating into your own code

Instead of using the hard-coded task in `main()`, you can import and drive the agent programmatically:

```python
from autonomous_coding_agent import CodingAgent, AgentConfig

config = AgentConfig(
    model_id="llama-3.3-70b-versatile",
    backup_dir=".agent_backups",
)
agent = CodingAgent(config=config)
result = agent.run("Your coding task description here", max_turns=5)
print(result)
```

The `task_description` should clearly describe what code to write or modify, desired behavior, and any constraints.

## Agent Protocol

The agent is instructed via its system prompt to:

- Think step-by-step in plain text.  
- Emit one or more ```actions``` blocks where each block is a YAML list of tool calls, for example:

  ```text
  ```actions
  - tool: write_file
    args:
      file_path: demo.py
      content: |
        print(42)
  ```
  ```

- End with exactly one ```yaml``` block declaring the outcome:

  ```text
  ```yaml
  outcome: success
  ```
  ```

The runtime parses each ```actions``` block, calls the corresponding methods (`list_directory_files`, `read_file`, `write_file`, etc.), aggregates observations, and feeds them back to the model on subsequent turns.

## Logging and Backups

- All stderr from Python file execution is appended to `execution_traceback.log` along with timestamps and return codes.
- Each time a file is written, a versioned backup is stored in `.agent_backups` (e.g., `foo.py_v1.bak`, `foo.py_v2.bak`, ...).
- You can revert a file to the latest backup using the `undo_last_change` tool.

## Limitations and Notes

- The agent currently focuses on Python projects and executes Python code only. 
- The filesystem root for operations is the current working directory; common directories such as `.git`, `__pycache__`, and `node_modules` are ignored.
- Long-running or hanging programs are bounded by configurable timeouts for both code execution and package installation.
```
