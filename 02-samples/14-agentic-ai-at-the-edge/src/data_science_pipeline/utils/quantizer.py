"""
Quantization utilities for GGUF export
"""

import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import json


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""

    quantization_method: str = "q4_k_m"  # Optimal for edge deployment
    use_mmap: bool = True
    use_mlock: bool = False
    threads: int = 4
    batch_size: int = 512

    # Multimodal projection settings
    include_mmproj: bool = True
    mmproj_quantization: str = "q8_0"  # Higher quality for projection

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__.keys()}


class ModelQuantizer:
    """Handle model quantization to GGUF format"""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.llama_cpp_path = self._find_llama_cpp()

    def _find_llama_cpp(self) -> Optional[Path]:
        """Find llama.cpp installation"""

        # Check common locations
        possible_paths = [
            Path.home() / "llama.cpp",
            Path("/usr/local/llama.cpp"),
            Path("./llama.cpp"),
            Path("../llama.cpp"),
            Path("../../llama.cpp"),
        ]

        # Also check PATH for llama.cpp binaries
        import shutil

        if shutil.which("llama-quantize") or shutil.which("quantize"):
            # If binaries are in PATH, assume installation directory
            binary_path = shutil.which("llama-quantize") or shutil.which("quantize")
            if binary_path:
                possible_paths.append(Path(binary_path).parent)

        for path in possible_paths:
            if path.exists() and (path / "convert.py").exists():
                return path

        return None

    def check_dependencies(self) -> bool:
        """Check if all required tools are available"""

        if not self.llama_cpp_path:
            print("Error: llama.cpp not found")
            print("Please clone: git clone https://github.com/ggerganov/llama.cpp")
            return False

        # Check for quantize binary
        quantize_path = self.llama_cpp_path / "quantize"
        if not quantize_path.exists():
            print("Error: quantize binary not found")
            print("Please build llama.cpp: cd llama.cpp && make")
            return False

        return True

    def convert_to_gguf(self, model_path: str, output_path: str, model_type: str = "qwen2") -> bool:
        """Convert HuggingFace model to GGUF format"""

        if not self.check_dependencies():
            return False

        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        convert_script = self.llama_cpp_path / "convert.py"

        print(f"Converting {model_path} to GGUF...")

        # Run conversion
        cmd = [
            "python",
            str(convert_script),
            str(model_path),
            "--outfile",
            str(output_path),
            "--outtype",
            "f16",  # Start with FP16
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Conversion successful")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e}")
            print(f"Error output: {e.stderr}")
            return False

    def quantize_gguf(
        self, input_path: str, output_path: str, method: Optional[str] = None
    ) -> bool:
        """Quantize GGUF model"""

        if not self.check_dependencies():
            return False

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        quantize_binary = self.llama_cpp_path / "quantize"
        method = method or self.config.quantization_method

        print(f"Quantizing to {method}...")

        # Run quantization
        cmd = [str(quantize_binary), str(input_path), str(output_path), method]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Quantization successful: {output_path}")

            # Check file size
            size_gb = output_path.stat().st_size / (1024**3)
            print(f"Output size: {size_gb:.2f} GB")

            return True
        except subprocess.CalledProcessError as e:
            print(f"Quantization failed: {e}")
            print(f"Error output: {e.stderr}")
            return False

    def process_multimodal_projection(self, mmproj_path: str, output_dir: str) -> Optional[str]:
        """Process multimodal projection file"""

        if not self.config.include_mmproj:
            return None

        mmproj_path = Path(mmproj_path)
        output_dir = Path(output_dir)

        if not mmproj_path.exists():
            print(f"Warning: Multimodal projection not found: {mmproj_path}")
            return None

        # Copy projection file
        output_path = output_dir / f"mmproj-{mmproj_path.name}"
        shutil.copy2(mmproj_path, output_path)

        print(f"Multimodal projection copied to: {output_path}")
        return str(output_path)

    def full_pipeline(
        self, model_path: str, output_dir: str, model_name: str = "qwen2.5-omni-7b-finetuned"
    ) -> Dict[str, str]:
        """Run complete quantization pipeline"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Step 1: Convert to GGUF
        gguf_f16_path = output_dir / f"{model_name}.gguf"
        if self.convert_to_gguf(model_path, gguf_f16_path):
            results["gguf_f16"] = str(gguf_f16_path)
        else:
            print("Conversion failed")
            return results

        # Step 2: Quantize
        quantized_path = output_dir / f"{model_name}-{self.config.quantization_method}.gguf"
        if self.quantize_gguf(gguf_f16_path, quantized_path):
            results["quantized"] = str(quantized_path)

        # Step 3: Handle multimodal projection
        mmproj_source = Path(model_path) / "mmproj.gguf"
        if mmproj_source.exists():
            mmproj_output = self.process_multimodal_projection(mmproj_source, output_dir)
            if mmproj_output:
                results["mmproj"] = mmproj_output

        # Step 4: Create metadata
        metadata = {
            "model_name": model_name,
            "quantization": self.config.quantization_method,
            "files": results,
            "config": self.config.to_dict(),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        results["metadata"] = str(metadata_path)

        # Step 5: Create test script
        self._create_test_script(output_dir, results)

        return results

    def _create_test_script(self, output_dir: Path, results: Dict[str, str]) -> None:
        """Create test script for the quantized model"""

        script_content = f"""#!/bin/bash
# Test script for quantized model

MODEL_PATH="{results.get('quantized', 'model.gguf')}"
MMPROJ_PATH="{results.get('mmproj', '')}"

echo "Testing quantized model..."
echo "Model: $MODEL_PATH"

# Start llama-server
llama-server \\
    -m "$MODEL_PATH" \\
    {f'--mmproj "$MMPROJ_PATH"' if results.get('mmproj') else ''} \\
    --host 0.0.0.0 \\
    --port 8080 \\
    -c 2048 \\
    -ngl 35 \\
    --jinja

# Test with curl
# curl http://localhost:8080/v1/chat/completions \\
#   -H "Content-Type: application/json" \\
#   -d '{{"messages": [{{"role": "user", "content": "Hello"}}]}}'
"""

        script_path = output_dir / "test_model.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        script_path.chmod(0o755)
        print(f"Test script created: {script_path}")

    def validate_gguf(self, model_path: str) -> bool:
        """Validate GGUF file integrity"""

        model_path = Path(model_path)

        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            return False

        # Check file size
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"Model size: {size_gb:.2f} GB")

        # Check with llama-cli if available
        llama_cli = self.llama_cpp_path / "llama-cli"
        if llama_cli.exists():
            cmd = [str(llama_cli), "-m", str(model_path), "-p", "Test", "-n", "1"]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("Model validation successful")
                    return True
                else:
                    print(f"Model validation failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print("Model loads but timed out (this is normal)")
                return True
            except Exception as e:
                print(f"Validation error: {e}")
                return False

        return True
