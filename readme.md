## Project Title: Skynet — A Self-Improving, Modular AGI Architecture

### Submitted by: \[Redacted for Application]

### Purpose

This document outlines the design, objectives, and ethical considerations of "Skynet," a student-led autonomous general intelligence system designed for educational and demonstrative purposes. It combines the GPT-OSS 20B open-weight model, recursive self-optimization, memory persistence, and system reflection to emulate characteristics of general intelligence. It is deployed in a sandboxed environment on both consumer and cloud infrastructure.

---

## 1. Introduction

Skynet is an experimental autonomous agent framework intended to simulate AGI-like capabilities using open-source tooling and commodity hardware. The system's name is tongue-in-cheek and satirical, referencing fictional AI with a dark reputation while explicitly working within sandboxed, safe, and ethical parameters. The purpose of this project is to explore cognitive architecture, system reflection, distributed intelligence, and autonomous tool use, while pushing the boundaries of what's possible with minimal resources.

## 2. Hardware and Infrastructure

* **Local Dev**: HP Z440 Workstation (Intel Xeon E5-2698 v3, 64 GB RAM)
* **Cloud Runtime**: Oracle Free Tier ARM instance (4 cores, 24 GB RAM, 200 GB SSD, 4 Gbps)
* **Experimental Platform**: Acer Chromebook Spin 713 (Intel i5-10210U)

## 3. Architectural Overview

Skynet's architecture consists of the following modular components:

* **Cognitive Core**: GPT-OSS 20B open-weight model
* **Memory Module**: Local vector DB w/ sentence-transformer embeddings + time-context indexing
* **Planner Module**: Goal decomposition using iterative chain-of-thought planning
* **Reflection Loop**: Recursively self-analyzes output logs and strategies
* **Critic Subsystem**: Evaluates accuracy, redundancy, or hallucinations
* **Execution Engine**: Calls Python tools, APIs, and external processes
* **Self-Editor**: Modifies its own prompts and architecture metadata
* **Subagent Orchestration**: Spawns specialized agents using task threading

## 4. Software Stack

* **Model Serving**: Local inference server for GPT-OSS 20B (via Hugging Face transformers / vLLM)
* **Orchestration**: CrewAI, Microsoft AutoGen, LangGraph
* **Memory/Storage**: SQLite/Chroma DB
* **Interface**: CLI, optional Web UI (Flask)
* **Security**: Sandboxed access, no unmonitored web output, ACL-based file access

## 5. Capabilities

* Autonomous planning and goal-setting
* Recursive improvement through prompt self-modification
* Long- and short-term memory embedding and retrieval
* Task delegation to generated subagents
* Reflexive analysis and runtime behavior tuning
* API interaction and environment introspection

## 6. Educational Value

* Demonstrates complexity of building general intelligence under constraint
* Explores philosophical and ethical boundaries of agency and autonomy
* Introduces concepts like memory persistence, reflexion, self-critique
* Encourages experimentation in a controlled, responsible format

## 7. Documentation & Deployment

* Hosted with dynamic DNS (No-IP) via `skynet.ddns.net`
* Available in open-source form (pending code audit)
* Deployment script includes environment lock, API limits, and kill-switch

## 8. Project Motivation

This project was born from curiosity about AGI architecture, and inspired by both fictional AI entities and modern open-weight model developments. The goal was never to create a dangerous agent but to demonstrate capability, foresight, responsibility, and ambition in designing autonomous systems.

## 9. Outcome & Future Vision

* Successful simulations of agent recursion and tool use
* Open framework to adapt to future open-weight LLMs (Mixtral, Gemma, etc.)
* Potential use in controlled, research-oriented simulation environments
* Demonstrates generalist reasoning without massive compute

## 10. Conclusion

Skynet (educational variant) represents an experiment in AGI architecture under strict control. Built with accessible tools, designed for safe execution, and aimed at furthering knowledge — not domination. Its very name challenges the fear around AGI by proving that powerful systems can be developed ethically, transparently, and with complete student ownership.

This work demonstrates a profound commitment to independent research, creative engineering, and self-motivated technical exploration.

---

**Appendices**

* A: Full agent config JSON schema
* B: Self-edit loop pseudocode
* C: Sample prompt trace from reflection module
* D: Screenshot of deployment on Oracle cloud
* E: GPT-OSS-generated ethical scenario simulation log

---

**End of Document (condensed)** — Full technical logs, metrics, and source code available upon request.
