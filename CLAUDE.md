# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Skynet is a modular, educational AGI architecture featuring multi-agent coordination, contextual memory, recursive self-improvement, and comprehensive auditing. It targets controlled, sandboxed deployments on commodity hardware.

## Key Architecture Components

- **Cognitive Engine**: Transformers-based reasoning with multiple modes
- **Contextual Memory**: Vector store + graph connections with decay and clustering
- **Multi-Agent Orchestrator**: Roles for Researcher, Planner, Executor, Critic, Reflector
- **Self-Editor**: Proposes and applies safe prompt/param/memory strategy edits
- **Audit System**: Full event logging, metrics, and integrity signatures
- **Tool System**: Minimal sandboxed runner for whitelisted commands

## Technology Stack

- **Model Serving**: Hugging Face Transformers with bitsandbytes/accelerate optimizations
- **Orchestration**: Custom orchestrator (optional integrations with CrewAI/AutoGen/LangGraph)
- **Memory/Storage**: ChromaDB + SQLite with embeddings via SentenceTransformers
- **Interface**: CLI (primary), optional Web UI
- **Deployment**: Local workstation and Oracle Free Tier ARM64 compatible

## Development Environment

- **Local Dev**: HP Z440 (Xeon E5-2698 v3, 64 GB RAM)
- **Cloud Runtime**: Oracle Free Tier ARM (4 cores, 24 GB RAM)
- **Testing**: Acer Chromebook Spin 713 (i5-10210U)

## Safety and Ethics

- Sandboxed execution enforced; allowed directories controlled via config
- Comprehensive auditing of memory, tools, self-edits, and decisions
- Human oversight required by default (Supervised safety level)
- Recursive self-improvement allowed with logging and rollback
- No unbounded autonomy; safety limits and kill-switches in config

## Common Development Tasks

1. Ensure `deploy.py` runs cleanly (deps, dirs, config)
2. Validate config via `SkynetConfig.validate_config()` before changes
3. Add features behind safety flags and log all operations
4. Keep memory/tool usage auditable and within sandbox paths
5. Favor small, composable modules and minimal external coupling

## Key Directories

- `/core/` – Cognitive engine and self-editor
- `/memory/` – Contextual memory system
- `/agents/` – Multi-agent orchestration
- `/tools/` – Sandboxed tool runner
- `/audit/` – Audit system and logs
- `/config/` – Configuration management

## Use Cases

- Educational demos of AGI concepts
- Research on multi-agent reasoning and memory systems
- Safe experimentation with recursive self-improvement

## Deployment

- Dynamic DNS as noted in `readme.md` (skynet.ddns.net)
- Educational-use focus (see README)
- Optimized for E5-2698v3 + K2200 and ARM64 cloud
