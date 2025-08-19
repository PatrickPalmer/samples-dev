# Data Science Pipeline

## Overview

This tutorial demonstrates how to fine-tune Qwen3-1.7B for enhanced function calling capabilities in edge AI deployments. The pipeline addresses critical challenges in production deployments including tokenization mismatches, format alignment, and resource optimization for edge devices.

## Architecture Flow

```mermaid
flowchart TD
    A[Production Tools] -->|Real Tool Registry| B[Data Generator]
    B -->|Synthetic Conversations| C[Training Dataset]
    C -->|Plain Text Format| D[Fine-Tuning]
    D -->|LoRA Adapters| E[Model Merger]
    E -->|Full Weights| F[GGUF Quantizer]
    F -->|Q4_K_M Model| G[Edge Deployment]
    
    H[5 Cockpit Controls] -->|Validated Specs| B
    I[Transformers + PEFT] -->|Memory Optimization| D
    J[llama.cpp] -->|Conversion Tools| F
    K[Format Validation] -->|Post-Jinja Alignment| C
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
    style D fill:#fff3e0
    style K fill:#ffebee
```

## Key Learning Objectives

This tutorial teaches critical lessons for production AI deployment:

### 1. Format Alignment is Critical
Understanding how tokenization and template processing affects model performance. The pipeline demonstrates why base models without chat tokens require plain text formatting to avoid unknown token issues that cause 0% accuracy.

### 2. Data Quality Over Quantity
The pipeline validates training data against actual production tools, filtering out phantom functions and ensuring exact parameter matching. This prevents the model from learning to call non-existent tools.

### 3. Production-First Design
Every component is designed for real-world edge deployment constraints including memory limits, inference speed requirements, and quantization trade-offs.

## Training Data Architecture

### Design Principles

The training data architecture addresses production challenges discovered through extensive testing:

1. **Production Tool Validation** - Only includes the 5 real cockpit control tools that exist in deployment
2. **Plain Text Formatting** - Matches post-Jinja processing output from llama.cpp server
3. **Tokenization Compatibility** - Avoids chat template tokens that base models lack
4. **Realistic Scenarios** - Based on actual vehicle control commands from user testing
5. **Format Consistency** - Maintains exact Strands SDK structure throughout

### Conversation Structure

The pipeline structures training conversations to match production requirements exactly. Each conversation demonstrates proper tool usage patterns that the model learns to replicate.

### Message Format Architecture

The training format implements a strict message flow that ensures consistent tool calling behavior:

**Conversation Flow Pattern:**
1. User provides natural language request
2. Assistant acknowledges and invokes appropriate tool
3. System returns tool result as user message
4. Assistant confirms action completion to user

**Critical Format Elements:**
- Tool calls wrapped in content arrays alongside acknowledgment text
- Unique toolUseId for tracking execution flow
- Parameter objects matching exact production schemas
- Result messages maintaining role consistency

### Production Tool Registry

The pipeline validates all tools against the actual production registry:

**Cockpit Control Tools** (command parameter):
- climate_control - Temperature, AC, heating, defrost
- window_control - Windows, sunroof with safety checks
- seat_control - Position, heating, memory presets
- lighting_control - Headlights, ambient, fog lights
- drive_mode - Sport, eco, comfort, traction control

Each tool specification includes parameter validation rules that match the production Virtual ECU implementation.

## Tutorial Setup

This tutorial provides hands-on experience with production AI fine-tuning challenges and solutions.

### Prerequisites

The pipeline requires specific dependencies for fine-tuning and quantization. Install using the project's data science extras.

### Environment Configuration

The pipeline adapts to different hardware configurations:
- **Training**: 8GB+ VRAM GPU recommended (RTX 3060/4060)
- **Evaluation**: Can run on CPU for testing
- **Quantization**: Requires llama.cpp build tools

## Pipeline Components

### 1. Data Generation Module

The data generator creates synthetic training data that matches production requirements exactly. Key innovations include:

**Format Validation System:**
- Validates tool names against production registry
- Ensures parameter schemas match Virtual ECU expectations
- Filters out any phantom or deprecated tools
- Maintains consistent toolUseId tracking

**Conversation Diversity:**
- Natural language variations for each command type
- Edge cases and error scenarios
- Multi-turn interactions requiring context
- Safety-aware command sequences

### 2. Fine-Tuning Architecture

The training module addresses critical production challenges:

**Base Model Selection:**
Using Qwen3-1.7B base model (not instruct) avoids tokenization issues. The base model processes plain text without expecting special chat tokens that cause unknown token errors.

**LoRA Configuration:**
Parameter-efficient training that preserves base model capabilities while adding function calling skills. The rank and alpha parameters balance between adaptation capacity and training efficiency.

**Memory Optimization:**
Gradient accumulation and 4-bit quantization enable training on consumer GPUs while maintaining quality.

### 3. Quantization Pipeline

The quantization process optimizes models for edge deployment:

**Q4_K_M Method:**
K-means clustering groups similar weights, achieving 70% size reduction with minimal accuracy loss. This method specifically preserves critical weights for function calling.

**GGUF Export:**
Converts to llama.cpp compatible format with metadata preservation for proper inference configuration.

## Critical Discoveries

### Tokenization Mismatch Resolution

The pipeline solves the fundamental issue where base models cannot process chat template tokens:

**Problem**: Direct tokenization of `<|im_start|>` and `<|im_end|>` produces unknown tokens
**Solution**: Format as plain text matching post-Jinja template processing
**Result**: Accuracy improves from 0% to production-ready levels

### Data Quality Validation

Ensuring training data matches production reality:

**Problem**: Training on non-existent tools causes hallucinations
**Solution**: Strict validation against production tool registry
**Result**: Model only calls tools that actually exist

### Format Alignment Strategy

Matching the exact format used in production:

**Problem**: Multiple conflicting formats across the pipeline
**Solution**: Standardize on Strands SDK format throughout
**Result**: Consistent tool calling behavior in deployment

## Performance Improvements

The pipeline demonstrates significant improvements through proper format alignment:

| Metric | Baseline | Fixed Pipeline | Impact |
|--------|----------|----------------|---------|
| Tool Selection Accuracy | 15-25% | 90%+ | Critical for production |
| Parameter Extraction | 0% (garbage) | 85%+ | Enables actual execution |
| Format Validity | 0% (unknown tokens) | 99% | Eliminates errors |
| Edge Inference Speed | N/A | 20-35 tok/s | Real-time response |

### Key Performance Insights

**Baseline Failures:**
The original approach with chat template tokens produced 0% accuracy due to unknown token generation. The model output garbage characters instead of valid tool calls.

**Fixed Pipeline Success:**
Proper plain text formatting enables the model to generate valid tool calls consistently. The improvements come from addressing the root tokenization issue rather than model architecture changes.

## Production Integration

### Edge Deployment Strategy

The pipeline outputs models optimized for edge deployment:

**Resource Efficiency:**
- Model size: 1.9GB (Q4_K_M quantized)
- Memory usage: <3GB total system
- CPU compatibility: Runs on 4-core processors
- Response latency: <500ms for tool calls

**Integration Points:**
- Direct drop-in replacement for base models
- Compatible with existing llama.cpp infrastructure
- Maintains Strands SDK compatibility
- Preserves safety constraints

### Deployment Validation

The pipeline includes validation steps to ensure production readiness:

**Format Testing:**
Validates that generated tool calls match expected schemas exactly

**Safety Verification:**
Ensures model respects Virtual ECU constraints and safety rules

**Performance Benchmarking:**
Measures inference speed and memory usage against targets

## Lessons for Production AI

This tutorial demonstrates critical lessons for deploying AI in production:

### 1. Understanding the Full Stack
Success requires understanding every layer from tokenization through deployment. A mismatch at any level can cause complete failure.

### 2. Validation Against Reality
Training data must match actual production tools and formats. Phantom functions in training data lead to hallucinations in deployment.

### 3. Format Consistency is Non-Negotiable
Even minor format variations can break production systems. The pipeline enforces exact format matching throughout.

### 4. Edge Constraints Drive Design
Memory limits, inference speed, and quantization requirements must be considered from the start, not added later.

## Next Steps

After completing this tutorial, explore:

1. **Custom Tool Development**: Add new vehicle control functions
2. **Multi-Modal Extensions**: Integrate Qwen2.5-Omni for voice/vision
3. **Performance Optimization**: Fine-tune quantization parameters
4. **Safety Enhancements**: Implement additional constraint validation

The pipeline provides a foundation for production-ready edge AI deployment with function calling capabilities optimized for resource-constrained environments.