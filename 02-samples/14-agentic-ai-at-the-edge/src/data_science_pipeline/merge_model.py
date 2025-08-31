#!/usr/bin/env python3
"""
Merge LoRA adapters with base Whisper model for whisper.cpp conversion.

This script merges the fine-tuned LoRA adapters with the base Whisper model
to create a single merged model that can be converted to GGML format for
use with whisper.cpp.
"""

import os
import sys
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import argparse
from pathlib import Path


def merge_lora_model(
    base_model_name: str = "openai/whisper-base",
    adapter_path: str = "whisper-japanese-finetuned",
    output_path: str = "whisper-japanese-finetuned-merged",
    device: str = "auto"
):
    """
    Merge LoRA adapters with base Whisper model.
    
    Args:
        base_model_name: Name or path of the base Whisper model
        adapter_path: Path to the LoRA adapter files
        output_path: Path where merged model will be saved
        device: Device to use for loading models
    """
    
    print(f"🔄 Starting model merge process...")
    print(f"📁 Base model: {base_model_name}")
    print(f"📁 Adapter path: {adapter_path}")
    print(f"📁 Output path: {output_path}")
    
    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load base model
        print("\n📥 Loading base model...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=device if device != "cpu" else None
        )
        print(f"   ✅ Base model loaded: {base_model_name}")
        
        # Load processor (tokenizer + feature extractor)
        print("\n📥 Loading processor...")
        processor = WhisperProcessor.from_pretrained(base_model_name)
        print(f"   ✅ Processor loaded")
        
        # Load and merge LoRA adapters
        print(f"\n🔗 Loading LoRA adapters from {adapter_path}...")
        model_with_adapters = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            torch_dtype=torch.float32
        )
        print(f"   ✅ LoRA adapters loaded")
        
        # Merge adapters into base model
        print("\n🔄 Merging LoRA adapters with base model...")
        merged_model = model_with_adapters.merge_and_unload()
        print(f"   ✅ Models merged successfully")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save merged model in PyTorch format for whisper.cpp compatibility
        print(f"\n💾 Saving merged model to {output_path}...")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=False,  # Save as PyTorch format for whisper.cpp
            max_shard_size="5GB"
        )
        print(f"   ✅ Model saved in PyTorch format")
        
        # Save processor
        print(f"\n💾 Saving processor to {output_path}...")
        processor.save_pretrained(output_path)
        print(f"   ✅ Processor saved")
        
        # Verify saved files
        print(f"\n🔍 Verifying saved files...")
        saved_files = list(Path(output_path).glob("*"))
        print(f"   📁 Saved files ({len(saved_files)} files):")
        for file in sorted(saved_files)[:10]:  # Show first 10 files
            print(f"      - {file.name}")
        if len(saved_files) > 10:
            print(f"      ... and {len(saved_files) - 10} more files")
        
        print(f"\n🎉 Model merge completed successfully!")
        print(f"📊 Summary:")
        print(f"   📁 Merged model saved to: {output_path}")
        print(f"   🔢 Total files: {len(saved_files)}")
        
        # Calculate model size
        total_size = sum(f.stat().st_size for f in Path(output_path).glob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"   📏 Total size: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model merge: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base Whisper model")
    parser.add_argument(
        "--base-model", 
        default="openai/whisper-base",
        help="Base model name or path (default: openai/whisper-base)"
    )
    parser.add_argument(
        "--adapter-path",
        default="whisper-japanese-finetuned", 
        help="Path to LoRA adapter files (default: whisper-japanese-finetuned)"
    )
    parser.add_argument(
        "--output-path",
        default="whisper-japanese-finetuned-merged",
        help="Output path for merged model (default: whisper-japanese-finetuned-merged)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Run merge
    success = merge_lora_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        device=args.device
    )
    
    if success:
        print(f"\n✅ Merge completed successfully!")
        print(f"💡 Next steps:")
        print(f"   1. Use whisper.cpp convert-pt-to-ggml.py to convert to GGML format")
        print(f"   2. Test the converted model with whisper.cpp")
        sys.exit(0)
    else:
        print(f"\n❌ Merge failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
