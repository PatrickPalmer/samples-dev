"""
Training utilities for Qwen3-1.7B fine-tuning with Strands SDK tool format
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    # Model configuration
    model_name: str = "Qwen/Qwen3-1.7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    # Training configuration
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Output configuration
    output_dir: str = "./models/fine-tuned"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 10

    # Hardware configuration
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {key: getattr(self, key) for key in self.__dataclass_fields__.keys()}

    def save(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ModelTrainer:
    """Trainer for Qwen3-1.7B model with Strands SDK tool calling"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Initialize model with Unsloth optimizations"""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed. Run: pip install unsloth")

        # Load model with Unsloth optimizations

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.config.load_in_4bit,
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if self.config.gradient_checkpointing else False,
            random_state=42,
        )

        # Track parameter counts for monitoring
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.param_ratio = 100 * trainable / total if total > 0 else 0

    def prepare_dataset(self, dataset_path: str):
        """Load and prepare training dataset"""
        from datasets import Dataset

        examples = []
        with open(dataset_path, "r") as f:
            for line in f:
                example = json.loads(line)
                formatted = self.format_conversation(example)
                examples.append({"text": formatted})

        dataset = Dataset.from_list(examples)

        # Split into train and validation
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        return train_dataset, eval_dataset

    def format_conversation(self, example: Dict[str, Any]) -> str:
        """Format conversation for training"""

        formatted = ""
        messages = example.get("messages", [])

        for message in messages:
            role = message["role"]

            if role == "system":
                formatted += f"<|im_start|>system\n{message['content']}<|im_end|>\n"

            elif role == "user":
                formatted += "<|im_start|>user\n"
                content = message["content"]

                if isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                formatted += item["text"]
                            elif "toolResult" in item:
                                result = item["toolResult"]
                                formatted += f"[Tool Result: {result['toolUseId']}]\n"
                                for content_item in result.get("content", []):
                                    if "text" in content_item:
                                        formatted += content_item["text"]
                            else:
                                # Handle other content types
                                content_type = item.get("type", "unknown")
                                formatted += f"[{content_type}]"
                        else:
                            formatted += str(item)
                else:
                    formatted += str(content)

                formatted += "<|im_end|>\n"

            elif role == "assistant":
                formatted += "<|im_start|>assistant\n"
                content = message["content"]

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                formatted += item["text"]
                            elif "toolUse" in item:
                                tool_use = item["toolUse"]
                                formatted += f"\n<tool_call>\n"
                                formatted += f"id: {tool_use['toolUseId']}\n"
                                formatted += f"name: {tool_use['name']}\n"
                                formatted += f"arguments: {json.dumps(tool_use['input'])}\n"
                                formatted += "</tool_call>\n"
                        else:
                            formatted += str(item)
                else:
                    formatted += str(content)

                formatted += "<|im_end|>\n"

        return formatted

    def train(self, train_dataset, eval_dataset):
        """Run training process"""
        import torch
        from transformers import TrainingArguments
        from trl import SFTTrainer

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            load_best_model_at_end=True,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
        )

        # Execute training
        self.trainer.train()

        # Save trained model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        return self.trainer.state.log_history

    def merge_and_save(self, output_path: str):
        """Merge LoRA weights and save full model"""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed")

        # Merge LoRA weights into base model
        merged_model = FastLanguageModel.merge_and_unload(self.model, save_method="merged_16bit")

        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return merged_model
