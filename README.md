# Skynet - Educational AGI Architecture

![Skynet Logo](https://img.shields.io/badge/Skynet-Educational%20AGI-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)
![License](https://img.shields.io/badge/License-Educational%20Use-orange?style=flat-square)

A modular, self-improving AGI architecture designed for educational and research purposes. Features recursive learning, multi-agent collaboration, and comprehensive audit trails.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- NVIDIA GPU with 4GB+ VRAM (optional but recommended)
- Linux/macOS/Windows

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd skynet
   python deploy.py
   ```

2. **Run interactive mode:**
   ```bash
   python skynet.py --interactive
   ```

3. **Process a single query:**
   ```bash
   python skynet.py --query "Explain quantum computing"
   ```

4. **Run autonomously:**
   ```bash
   python skynet.py --autonomous 30  # Run for 30 minutes
   ```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SKYNET CORE                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Cognitive      │  │  Memory System   │  │  Audit      │ │
│  │  Engine         │  │                  │  │  System     │ │
│  │  - GPT Model    │  │  - Vector DB     │  │  - Complete │ │
│  │  - Reasoning    │  │  - Context       │  │    Logging  │ │
│  │  - Multi-mode   │  │    Graphs        │  │  - Safety   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Self-Editor    │  │  Agent           │  │  Tool       │ │
│  │                 │  │  Orchestrator    │  │  System     │ │
│  │  - Auto         │  │  - Multi-agent   │  │  - Sandbox  │ │
│  │    Optimization │  │  - Collaboration │  │  - Safe     │ │
│  │  - Recursive    │  │  - Specialization│  │    Execution│ │
│  │    Learning     │  │                  │  │             │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Key Features

### Recursive Self-Improvement
- **Automatic Optimization**: System continuously analyzes its own performance
- **Parameter Tuning**: Dynamically adjusts model parameters based on results
- **Prompt Evolution**: Self-modifies reasoning templates for better outcomes
- **Full Audit Trail**: Every self-modification is logged and reversible

### Recursive Training (LoRA)
- **Experience Buffer**: Stores query/response pairs with rewards
- **Evaluator**: Computes rewards and aggregates metrics
- **LoRA Trainer**: Fine-tunes adapters on accumulated data
- **Scheduler**: Trains, evaluates, and promotes adapters when metrics improve
- **Safety Gating**: Honors human-oversight and audit logging

### Multi-Agent Collaboration
- **Specialized Agents**: Researcher, Planner, Executor, Critic, Reflector roles
- **Dynamic Task Distribution**: Automatic assignment based on agent capabilities
- **Collaborative Problem Solving**: Multiple agents work together on complex tasks
- **Inter-Agent Communication**: Message routing and coordination

### Contextual Memory System
- **Vector-Based Storage**: Semantic search using sentence transformers
- **Hierarchical Graphs**: Connected memory nodes with contextual relationships
- **Time-Aware Decay**: Automatic importance adjustment over time
- **Episodic Retrieval**: Context-aware memory reconstruction

### Comprehensive Auditing
- **Complete Transparency**: Every system operation is logged
- **Performance Tracking**: Detailed metrics and statistics
- **Safety Monitoring**: Real-time violation detection
- **Rollback Capability**: Restore previous system states

### Tool System (Sandboxed)
- **Safe Execution**: Minimal, whitelisted command runner
- **Sandbox Enforcement**: Honors allowed directories in config
- **Audit Hooks**: All tool invocations are logged

## ⚙️ Configuration

Edit `config/config.json` to customize system behavior:

```json
{
  "hardware": {
    "cpu_cores": 32,
    "max_memory_gb": 60,
    "gpu_enabled": true,
    "quantization_bits": 4
  },
  "safety": {
    "safety_level": "supervised",
    "human_oversight_required": true,
    "max_self_edits_per_hour": 100,
    "sandbox_enabled": true
  },
  "model": {
    "primary_model": "microsoft/DialoGPT-large",
    "temperature": 0.7,
    "max_context_length": 2048
  }
  ,
  "training": {
    "enabled": false,
    "min_batch_size": 64,
    "improvement_threshold": 0.02
  }
}
```

## 🛡️ Safety Features

### Multi-Level Safety System
- **Sandboxed Execution**: All operations run in isolated environment
- **Human Oversight**: Optional human approval for modifications
- **Emergency Shutdown**: Multiple kill switches and safety controls
- **Audit Compliance**: Complete logging for regulatory compliance

### Safety Levels
1. **SANDBOX_ONLY**: No external access, limited self-modification
2. **RESTRICTED**: Basic operations with human oversight
3. **SUPERVISED**: Advanced features with logging
4. **AUTONOMOUS**: Full capabilities (use with caution)

## 📊 Monitoring and Statistics

### Real-Time Monitoring
```bash
python skynet.py --stats
```

### Key Metrics
- **Performance**: Response time, confidence scores, success rates
- **Memory**: Storage efficiency, retrieval accuracy, graph connectivity
- **Agents**: Task distribution, collaboration effectiveness
- **Self-Improvement**: Modification success rate, performance gains

## 🔧 Hardware Optimization

### Optimized for E5-2698v3 + K2200
- **CPU Utilization**: Multi-threaded processing with thread pools
- **Memory Management**: Intelligent caching and cleanup
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Model Quantization**: 4-bit/8-bit quantization for memory efficiency

### Scalability Options
- **Cloud Deployment**: Oracle Free Tier ARM64 compatibility
- **Edge Computing**: Quantized models for resource-constrained devices
- **Distributed Processing**: Multi-node agent orchestration

## 📝 Usage Examples

### Interactive Chat
```bash
python skynet.py --interactive
Skynet> Explain the concept of recursive self-improvement
Skynet> What are the ethical implications of AGI?
Skynet> stats  # Show system statistics
```

### Autonomous Operation
```bash
# Run autonomously for 1 hour
python skynet.py --autonomous 60

# The system will:
# - Analyze its own performance
# - Generate improvement proposals  
# - Apply beneficial modifications
# - Expand knowledge through reflection
```

### Batch Processing
```python
from skynet import SkynetSystem
import asyncio

async def process_queries():
    skynet = SkynetSystem()
    await skynet.initialize()
    
    queries = [
        "Analyze current AI safety research trends",
        "Plan a comprehensive study of machine learning",
        "Reflect on the implications of recursive improvement"
    ]
    
    for query in queries:
        result = await skynet.process_query(query)
        print(f"Query: {query}")
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 50)
    
    await skynet.shutdown()

asyncio.run(process_queries())
```

## 🔍 Development and Research

### Extending Skynet

1. **Custom Agents**: Implement new agent roles for specific domains
2. **Tool Integration**: Add new tools and capabilities
3. **Memory Modules**: Extend memory systems with new storage types  
4. **Reasoning Modes**: Implement new reasoning strategies

### Research Applications
- **Cognitive Architecture**: Study multi-agent reasoning systems
- **Self-Improvement**: Research recursive optimization algorithms
- **Memory Systems**: Investigate contextual knowledge representation
- **AI Safety**: Develop and test safety mechanisms

## 📚 Documentation

### Core Components
- [`skynet.py`](skynet.py) - Main system orchestrator
- [`config/skynet_config.py`](config/skynet_config.py) - Configuration management
- [`audit/audit_system.py`](audit/audit_system.py) - Comprehensive logging
- [`memory/contextual_memory.py`](memory/contextual_memory.py) - Memory system
- [`core/cognitive_engine.py`](core/cognitive_engine.py) - Reasoning engine
- [`core/self_editor.py`](core/self_editor.py) - Self-improvement system
- [`agents/agent_orchestrator.py`](agents/agent_orchestrator.py) - Multi-agent coordination
- [`tools/sandbox_tools.py`](tools/sandbox_tools.py) - Sandboxed tool runner

### Logs and Auditing
- `audit/logs/` - All system operations and decisions
- `audit/logs/skynet_YYYYMMDD.log` - Daily operation logs
- `config/config.json` - Current system configuration

## ⚠️ Important Notes

### Educational Purpose
This system is designed for **educational and research purposes**. It demonstrates AGI concepts including:
- Recursive self-improvement
- Multi-agent collaboration  
- Contextual memory systems
- Comprehensive auditing

### Responsible Use
- **Always** review self-modifications before applying
- **Monitor** system behavior through audit logs
- **Use appropriate** safety settings for your environment
- **Understand** the implications of autonomous operation

### Limitations
- Requires substantial computational resources
- Performance depends on available hardware
- Educational implementation - not production ready
- Requires technical knowledge for advanced usage

## 🤝 Contributing

This is an educational project. Contributions should focus on:
- Safety improvements
- Educational value
- Code clarity and documentation
- Research applications

## 📄 License

This project is released for educational use only. See LICENSE for details.

## 🙋‍♂️ Support

For questions, issues, or research collaboration:
- Review the audit logs in `audit/logs/`
- Check configuration in `config/config.json`
- Monitor system statistics with `python skynet.py --stats`

---

**⚠️ Remember: This is a powerful system capable of self-modification. Always use appropriate safety measures and monitoring.**

*"The future belongs to those who understand both the potential and the responsibility of artificial intelligence."*
### AGI-Like Meta-Cognition
- **Global Workspace**: Shared salience-driven bus for cross-module attention
- **Self-Model**: Tracks calibration, competencies, blind spots
- **Intrinsic Goals**: Curiosity/competence goals scheduled with safety gating
