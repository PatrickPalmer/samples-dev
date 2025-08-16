# Fine-Tuning Pipeline for Qwen2.5-Omni

Pipeline for fine-tuning Qwen2.5-Omni-7B with tool calling capabilities for edge deployment.

## Overview

This pipeline enables data scientists and engineers to fine-tune large language models for tool calling in edge environments. The implementation uses Unsloth for efficient training on consumer GPUs (14GB VRAM) and produces quantized models optimized for deployment with llama.cpp.

## Quick Start

### Prerequisites

```bash
# Use the existing virtual environment
source /path/to/your/venv/bin/activate

# Install additional requirements
pip install jupyter notebook
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers>=4.52.3 datasets trl peft
```

### Run the Pipeline

```bash
# Navigate to pipeline directory
cd /path/to/14-agentic-ai-at-the-edge/src/data_science_pipeline

# Start Jupyter
jupyter notebook training.ipynb
```

## Pipeline Stages

### Stage 1: Data Generation
Generates synthetic training data matching the Strands SDK tool calling format.

**Key Features:**
- 10 different tool types (cockpit controls, assistants, multimodal)
- Realistic conversation patterns from production usage
- Proper Strands SDK format with `toolUse` blocks

**Output:** `data/train.jsonl`, `data/test.jsonl`

### Stage 2: Fine-Tuning
Trains the model using Unsloth with LoRA adapters.

**Configuration:**
- 4-bit quantization for 14GB VRAM compatibility
- LoRA rank 16, alpha 32
- Context window: 2048 tokens (matches deployment)
- Gradient checkpointing for memory efficiency

**Output:** `models/fine-tuned/` (LoRA adapters)

### Stage 3: Quantization
Converts to GGUF format for llama.cpp deployment.

**Process:**
1. Merge LoRA weights with base model
2. Convert to GGUF format
3. Apply Q4_K_M quantization
4. Include multimodal projection

**Output:** `outputs/gguf/qwen2.5-omni-finetuned-q4_k_m.gguf`

### Stage 4: Evaluation
Tests model performance on tool calling tasks.

**Metrics:**
- Tool selection accuracy (target: >95%)
- Parameter extraction accuracy (target: >90%)
- Inference performance (target: 20-35 tokens/s)
- Memory usage (target: <3GB)

**Output:** `outputs/evaluation_metrics.json`

## Directory Structure

```
data_science_pipeline/
├── README.md                    # This file
├── fine_tuning_pipeline.ipynb  # Main notebook
├── utils/                       # Helper modules
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic data creation
│   ├── trainer.py              # Unsloth training
│   ├── quantizer.py            # GGUF conversion
│   └── evaluator.py            # Performance testing
├── data/                        # Generated datasets
│   ├── train.jsonl
│   └── test.jsonl
├── models/                      # Model artifacts
│   ├── fine-tuned/             # LoRA adapters
│   └── merged/                 # Full models
└── outputs/                     # Final outputs
    ├── gguf/                   # Quantized models
    └── evaluation_metrics.json
```

## Tool Specifications

The pipeline trains the model to use these tools:

### Cockpit Controls
- `climate_control(command: str)` - Temperature, AC, defrost
- `window_control(command: str)` - Windows, sunroof
- `seat_control(command: str)` - Position, heating, memory
- `lighting_control(command: str)` - Headlights, ambient, fog
- `drive_mode(command: str)` - Sport, eco, traction

### Model Selection
- `select_model(query: str)` - Dynamic model routing based on complexity

## Training Data Format

The model learns to generate tool calls in this format:

```json
{
  "role": "assistant",
  "content": [
    {
      "text": "I'll help you with that."
    },
    {
      "toolUse": {
        "toolUseId": "call_abc123",
        "name": "climate_control",
        "input": {
          "command": "Set temperature to 72 degrees"
        }
      }
    }
  ]
}
```

## Validation

### Test Data Generation

```python
from utils import DataGenerator, ToolRegistry

# Generate test dataset
generator = DataGenerator()
generator.generate_dataset(
    num_examples=100,
    output_path="./data/validation.jsonl"
)
```

### Test Model Inference

```bash
# Start llama-server with fine-tuned model
llama-server \
  -m outputs/gguf/qwen2.5-omni-finetuned-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 2048 -ngl 35 --jinja

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Set the temperature to 72 degrees"}
    ],
    "max_tokens": 200
  }'
```

### Integration Test

```python
from strands.models.llamacpp import LlamaCppModel

# Connect to fine-tuned model
model = LlamaCppModel(
    base_url="http://localhost:8080",
    params={"temperature": 0.7, "max_tokens": 1024}
)

# Test tool calling
response = model.generate(
    messages=[{"role": "user", "content": "Open the driver window"}],
    tool_specs=[window_control_spec]
)
```

## Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Tool Selection Accuracy | >95% | TBD |
| Parameter Extraction | >90% | TBD |
| Inference Speed | 20-35 tok/s | TBD |
| Memory Usage | <3GB | TBD |
| First Token Latency | <200ms | TBD |

## Deployment

### Copy to Edge Device

```bash
# Copy quantized model to edge deployment
cp outputs/gguf/qwen2.5-omni-finetuned-q4_k_m.gguf \
   ../../edge/models/

# Update edge configuration
export MODEL_PATH=/app/models/qwen2.5-omni-finetuned-q4_k_m.gguf
```

### Docker Integration

```dockerfile
# In Dockerfile.edge
COPY models/qwen2.5-omni-finetuned-q4_k_m.gguf /app/models/
ENV MODEL_PATH=/app/models/qwen2.5-omni-finetuned-q4_k_m.gguf
```

## Troubleshooting

### Out of Memory

If training fails with OOM:
1. Reduce batch size to 1
2. Increase gradient accumulation steps
3. Enable CPU offloading
4. Use smaller sequence length

### Quantization Issues

If llama.cpp conversion fails:
1. Ensure llama.cpp is built: `cd llama.cpp && make`
2. Check model format compatibility
3. Try different quantization methods (q5_k_m, q8_0)

### Poor Tool Calling Accuracy

If the model doesn't call tools correctly:
1. Increase training epochs
2. Adjust learning rate (try 1e-4)
3. Add more diverse training examples
4. Check data format consistency

## Best Practices

1. **Data Quality**: Ensure training data exactly matches production format
2. **Validation**: Always test on held-out data before deployment
3. **Versioning**: Tag models with training date and metrics
4. **Monitoring**: Track inference performance in production
5. **Iteration**: Fine-tune based on real usage patterns

## Contributing

To improve the pipeline:

1. Add more tool types in `utils/data_generator.py`
2. Experiment with different LoRA configurations
3. Test alternative quantization methods
4. Share benchmark results

## License

This pipeline is part of the Strands SDK samples and follows the same licensing terms.