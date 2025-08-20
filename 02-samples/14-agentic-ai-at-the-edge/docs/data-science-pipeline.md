# Data Science Pipeline Documentation

## Overview

This document describes the data science pipelines that enable multimodal AI capabilities for edge deployment. The implementation encompasses two complementary systems: function calling fine-tuning for tool execution and automatic speech recognition fine-tuning for voice interaction. Together, these pipelines create a production-ready system capable of understanding spoken commands and executing appropriate actions through vehicle control interfaces.

## System Architecture

```mermaid
flowchart LR
    A[Voice Command] --> B[ASR Pipeline]
    B --> C[Text Transcription]
    C --> D[Function Calling Pipeline]
    D --> E[Tool Execution]
    E --> F[Vehicle Control]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style D fill:#f3e5f5
    style F fill:#e8f5e9
```

The data science pipeline implements a modular architecture designed for edge deployment constraints while maintaining high performance standards. The system processes voice input through a fine-tuned ASR model, interprets commands using a specialized function calling model, and executes actions through a Virtual ECU interface that ensures safety compliance.

Voice commands captured through device microphones undergo initial processing through the Whisper-based ASR pipeline. The transcribed text then flows to the function calling model, which maps natural language to specific tool invocations. This two-stage approach enables robust handling of diverse accents, languages, and speaking styles while maintaining precise control execution.

## Function Calling Pipeline

The function calling pipeline addresses fundamental challenges in deploying language models for tool execution on edge devices. Through careful engineering of training data formats and model architectures, the system achieves production-grade reliability while operating within strict resource constraints.

### Technical Implementation

The function calling pipeline employs Qwen3-1.7B as its foundation model, selected for its optimal balance between capability and resource efficiency. The base model undergoes fine-tuning using Low-Rank Adaptation (LoRA), which reduces trainable parameters to approximately 2% of the original model while maintaining full performance.

Training data generation leverages Claude 3.5 Sonnet via AWS Bedrock to create diverse, realistic conversations that cover the full spectrum of supported commands. Each generated example undergoes validation to ensure format compliance and tool existence. A critical innovation in the pipeline involves solving the tokenization mismatch problem that causes baseline models to fail completely. The training data uses Qwen3 chat template tokens that match the model's vocabulary, ensuring proper token recognition during inference.

Post-training quantization reduces model size using Q4_K_M quantization, which employs k-means clustering to group similar weights. This size reduction enables deployment on devices with limited storage while maintaining model quality. The quantized model exports to GGUF format for direct integration with llama.cpp inference servers.

## Automatic Speech Recognition Pipeline

The ASR pipeline fine-tunes OpenAI's Whisper model for specialized language support, with particular emphasis on Japanese speech recognition. This implementation demonstrates how foundation models trained on multilingual data can be efficiently adapted for specific language requirements in edge deployments.

### Language-Specific Challenges

Japanese ASR presents unique technical challenges that the pipeline addresses through specialized processing. The language uses three distinct writing systems often mixed within sentences, lacks natural word boundaries, and contains numerous homophones that require contextual disambiguation.

The pipeline implements Unicode NFKC normalization to handle full-width and half-width character variations consistently. Character Error Rate (CER) serves as the primary evaluation metric rather than Word Error Rate (WER), as it better reflects performance in continuous-script languages.

### Training Process

| Component | Specification |
|-----------|--------------|
| Base Model | Whisper-base (74M parameters) |
| Adaptation Method | LoRA on attention layers |
| Training Data | Mozilla Common Voice |
| Sample Size | 2000 utterances |
| Optimization | BF16 mixed precision |

The ASR pipeline utilizes Whisper-base as the foundation model, offering an optimal balance between accuracy and computational efficiency. LoRA adaptation targets the attention layers, significantly reducing trainable parameters while maintaining full model capabilities. Training implements gradient checkpointing and mixed-precision computation, enabling efficient processing on consumer GPUs.

## Integration with Production Systems

The complete inference pipeline orchestrates both ASR and function calling models to provide seamless voice-controlled tool execution. Audio input captured through the device microphone flows through the ASR model for transcription. The resulting text feeds into the function calling model, which determines the appropriate tool and parameters. Finally, the Virtual ECU executes the command while enforcing safety constraints.

This architecture maintains modularity, allowing independent updates to either model without affecting the overall system. The design also supports fallback mechanisms, reverting to text input if ASR confidence falls below acceptable thresholds.

### Production Tools

The system supports five primary vehicle control tools, each implementing natural language processing for command interpretation:

- **Climate Control**: Temperature, fan speed, and air distribution with safety limits
- **Window Control**: Individual or all windows with speed-based safety interlocks  
- **Seat Control**: Position and heating/cooling with movement restrictions while driving
- **Lighting Control**: Headlights and interior illumination with automatic activation
- **Drive Mode**: Sport, eco, and comfort settings with stationary vehicle requirements

All tools follow a consistent pattern of accepting natural language commands, parsing intent and parameters, validating against safety rules, and executing through the Virtual ECU interface.

## Testing and Validation

The pipeline includes extensive testing at multiple levels to ensure production reliability. Unit tests validate individual components including data generation, model loading, and inference. Integration tests verify end-to-end functionality from audio input to tool execution. Performance benchmarks measure latency, throughput, and resource utilization against requirements.

Safety testing confirms that all Virtual ECU constraints are properly enforced, preventing dangerous operations. Edge case testing evaluates system behavior with noisy audio, ambiguous commands, and out-of-vocabulary inputs.

Production deployments incorporate telemetry for ongoing performance monitoring. Metrics include inference latency percentiles, error rates by command type, and resource utilization patterns. This data drives continuous improvement through model updates and system optimization.

## Deployment Considerations

Training requires a CUDA-capable GPU with 8GB+ VRAM for efficient processing. An A10G or similar GPU completes training in 20-40 minutes depending on dataset size. For inference, models run on 4-core ARM or x86 processors with 4GB RAM minimum.

The system supports various deployment scenarios from automotive computing platforms to industrial IoT devices. Optimization for specific hardware accelerators like Neural Processing Units (NPUs) can further improve performance.

The pipeline builds on established open-source frameworks including PyTorch for model training, Transformers and PEFT for efficient fine-tuning, and llama.cpp for optimized inference. Complete dependency specifications in the project's pyproject.toml ensure reproducible environments.

## Getting Started

To begin using the pipelines:

1. Install the required dependencies from pyproject.toml
2. Open the appropriate Jupyter notebook for your use case
3. Follow the step-by-step instructions for data preparation and training
4. Export the trained model for edge deployment
5. Integrate with the production system using provided deployment scripts

## Conclusion

This data science pipeline demonstrates how modern AI techniques can be successfully adapted for edge deployment in production environments.