# Agentic AI at the Edge

An intelligent cockpit control system designed for edge deployment in resource constrained devices, featuring voice input powered by FFmpeg-Whisper integration, dynamic model routing with Qwen3-1.7B, and specialized agents for vehicle controls.

## Overview

This project demonstrates a **unified codebase architecture** that seamlessly runs across development, container, and production service environments without modification. The same `main.py` adapts its behavior based on deployment context, showcasing true write-once, deploy-anywhere capabilities.


### Core Features
- **Advanced AI reasoning** powered by Qwen3-1.7B with thinking/non-thinking mode capabilities
- **Voice-enabled processing** with FFmpeg-Whisper integration for optimal transcription
- **Cockpit controls** for climate, windows, seats, lights, and drive modes
- **Dynamic model routing** between local Qwen3 and cloud models based on query complexity  
- **Multilingual support** for 100+ languages with superior instruction following
- **Deployment-aware configuration** that automatically adapts to environment
- **Zero code changes** required between development and production

## Architecture

### System Overview

```mermaid
flowchart TB
    subgraph "Input Layer"
        A[User Input]
        B[Voice Input]
    end
    
    subgraph "Processing Layer"
        C[Model Selector]
        D[Orchestrator - Alex]
        E[Qwen3-1.7B Model]
        F[Bedrock Model]
    end
    
    subgraph "Agent Layer"
        G[Climate Control]
        H[Window Control]
        I[Seat Control]
        O[Lighting Control]
        P[Drive Mode]
    end
    
    subgraph "Resource Layer"
    end
    
    A --> D
    B --> E
    A --> C
    C --> E
    C --> F
    E --> D
    F --> D
    D --> G
    D --> H
    D --> I
    D --> O
    D --> P
```

### Request Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant ModelSelector
    participant Agent
    participant Knowledge
    
    User->>Orchestrator: Submit query
    Orchestrator->>ModelSelector: Analyze complexity
    ModelSelector-->>Orchestrator: Return model choice
    Orchestrator->>Orchestrator: Update model
    Orchestrator->>Agent: Route to control agent
    
    alt Cockpit Control
        Agent->>Agent: Execute command
        Agent-->>Orchestrator: Return status
    end
    
    Agent-->>Orchestrator: Return results
    Orchestrator-->>User: Formatted response
```

### Voice Processing Flow

```mermaid
flowchart LR
    A[Voice Command] --> B{Audio Support?}
    B -->|Yes| C[Record Audio]
    B -->|No| D[Text Fallback]
    C --> E[FFmpeg Whisper]
    E --> F[Text Transcription]
    F --> G[Qwen3-1.7B Processing]
    D --> G
    G --> H[Intelligent Response]
```

## Deployment Modes

### 1. Development Mode
Direct execution on developer machine with full capabilities:
```bash
python main.py
```
- **Audio**: Direct microphone access via sounddevice
- **Models**: Dynamic selection between local/cloud
- **Interface**: Rich terminal UI with color output
- **Use Case**: Development, testing, demonstrations

### 2. Container Mode  
Dockerized deployment for edge devices:
```bash
docker run -v $(pwd)/audio_exchange:/app/audio_exchange agentic-ai-edge
```
- **Audio**: File-based exchange through volume mount
- **Models**: Prioritizes local models for offline operation
- **Interface**: Terminal interface inside container
- **Use Case**: Edge devices, automotive systems, IoT

### 3. Service/API Mode
RESTful service for integration:
```bash
ENABLE_API=true python main.py
# or
docker run -e ENABLE_API=true -p 8000:8000 agentic-ai-edge
```
- **Audio**: Base64-encoded in JSON payload
- **Models**: Deployment-aware selection
- **Interface**: HTTP REST API with session support
- **Use Case**: Android Auto, web services, microservices

## Quick Start

### Prerequisites

1. Python 3.8 or higher
2. Strands SDK: `pip install git+https://github.com/westonbrown/sdk-python.git@main`
3. llama.cpp with server support
4. FFmpeg with Whisper support (compiled in container)
5. (Optional) AWS credentials for Bedrock access

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
python main.py

# Or run in API mode
ENABLE_API=true python main.py
```

## Edge Deployment Architecture

### The Power of Unified Codebase

The edge deployment exemplifies the architectural principle of **"write once, deploy anywhere"**. The exact same `main.py` that runs on a developer's laptop also powers automotive systems, IoT devices, and cloud services.

```
Developer Laptop          Edge Device              Production Service
    main.py      ═══>      main.py       ═══>         main.py
       ↓                      ↓                          ↓
  [Dev Mode]            [Container Mode]            [API Mode]
```

### How It Works

1. **Environment Detection**: The application detects its deployment context through environment variables
2. **Automatic Adaptation**: 
   - Audio input method switches automatically (mic → file → API)
   - Model selection adapts (cloud-preferred → local-only)
   - Interface changes (rich UI → container UI → REST API)
3. **Zero Code Changes**: Deploy the same code to a car, robot, or cloud server

### Quick Start
```bash
cd src/edge/deployment
./setup.sh  # Downloads models and starts container
```

### Real-World Example: Automotive Integration

```bash
# Same codebase, different environment variable
ENABLE_API=true DEPLOYMENT_TARGET=automotive ./setup.sh

# Android Auto sends voice command
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "voice",
    "audio_data": "<base64_encoded_audio>",
    "session_id": "driver_123"
  }'

# Response uses local models, accesses offline vehicle data
{
  "response": "The tire pressure warning indicates...",
  "session_id": "driver_123"
}
```

### Model Selection Logic

```mermaid
flowchart TD
    A[User Query] --> B{Analyze Query}
    B --> C{Complex?}
    C -->|Yes| D[Check Cloud Available]
    C -->|No| E[Use Local Model]
    D --> F{Cloud OK?}
    F -->|Yes| G[Use Bedrock]
    F -->|No| E
    E --> H[LlamaCpp Response]
    G --> I[Bedrock Response]
```

## Testing

```bash
cd tests
python test_all.py
```

## Architectural Summary

This project demonstrates that sophisticated AI systems don't require separate codebases for different deployment targets. By designing with deployment flexibility in mind, we achieve:

### Single Source of Truth
- **One `main.py`** serves all deployment scenarios
- **One set of agents** works everywhere  
- **One audio system** adapts to available inputs
- **One configuration** responds to environment

### Deployment Flexibility
```python
# The same code runs in all these scenarios:

# Developer's laptop
$ python main.py

# Automotive edge device  
$ docker run -e DEPLOYMENT_TARGET=automotive ...

# Cloud API service
$ ENABLE_API=true python main.py

# Each automatically adapts its behavior to the environment
```

## License

This project is part of the Strands SDK samples collection.
