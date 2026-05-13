This project is an AI-powered coding agent that can plan, write, execute, debug, and validate code across Python 
and C/C++ tasks. It includes multi-backend LLM support, semantic memory for reusable solutions, file 
backup/revert utilities, and a test suite for evaluating agent behavior.

Overview

The program is centered around a `CodingAgent` that takes a task prompt, creates a plan, executes tool actions such 
as writing files or running scripts, and checks results against expected output files and validation rules. It is 
designed to automate end-to-end coding workflows rather than just generate code text.

Key capabilities include:

Multi-backend LLM orchestration with Groq, OpenRouter, and Google Gemini support.
Semantic memory for reusing similar past solutions and learning from failures.
Tool execution for file operations, shell commands, Python scripts, and C++ compilation.
Automatic validation against required files and required output substrings.

Features

LLM backend management

The agent supports multiple API-backed providers and falls back between them when one is unavailable or rate-limited. 
It also includes caching and cooldown logic to reduce repeated calls.

Semantic memory

The memory system stores successful solutions and failure patterns, using TF-IDF-style text matching and hybrid 
similarity scoring to retrieve relevant prior tasks. This helps the agent reuse working plans and learn from 
recurring errors.

Tool execution

The tool executor can:

write and read files,
create directories,
execute Python scripts,
compile and run C++ files,
run shell commands,
install packages.

It also validates file syntax before writing and can auto-execute generated scripts when validation requires 
runtime output.

File safety and backups

The file manager creates backups before overwriting files, can revert to previous versions, and automatically fixes 
common include issues in C++ files.

Test suite

The included test suite contains a range of coding tasks such as concurrency, SQLite usage, numerical methods, file refactoring, data processing, and secure input validation. These are used to exercise different agent capabilities.

Requirements

The program expects:

Python 3.
API keys for at least one supported backend in environment variables such as `GROQAPIKEY`, `OPENROUTERAPIKEY`, or `GOOGLEAPIKEY`. Optional native build tools for C++ compilation, plus any system libraries needed by tasks such as `curl` or OpenSSL.

Usage

Run the program and choose one of the interactive options:

1. Single task mode.
2. Run test suite.
3. Print memory stats.
4. Quit.

In single task mode, the program prompts for a task description, maximum turns, expected output files, and optional validation checks, then executes the agent workflow automatically.

Example workflow

A typical run is:

1. Enter a task description.
2. Specify output files such as `app.py` or `output.json`.
3. Add validation requirements if needed.
4. Let the agent plan, write, execute, and verify the solution.

Configuration

Configuration is handled through the `AgentConfig` dataclass. Important settings include:

execution and code timeouts,
memory database path,
backup directory,
similarity threshold,
maximum memory entries,
backend priority order,
global rate limits.

Defaults are loaded from environment-based configuration, so the agent can be adapted without code changes.

Project structure

The main components are:

`AgentConfig`: runtime configuration.
`LLMBackendManager`: backend selection and API routing.
`SemanticMemoryManager`: solution and failure memory.
`FileManager`: safe file writing and backups.
`CodeExecutor`: runs Python/C++ code and shell commands.
`ToolExecutor`: validates and dispatches tool actions.
`CodingAgent`: the main task execution loop.
`TestSuite`: predefined tasks for evaluation.

Notes

The agent is intentionally strict about validation. It expects exact output files, exact substrings in program output, and can stop early if a task fails validation.
