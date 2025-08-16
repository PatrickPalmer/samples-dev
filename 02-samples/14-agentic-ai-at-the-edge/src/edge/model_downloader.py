#!/usr/bin/env python3
"""
Model Downloader for Edge Deployment

Automatically downloads Qwen2.5-Omni model and multimodal projection files
for voice-enabled personal assistant edge deployment.
"""

import os
import urllib.request
from pathlib import Path
import hashlib
import sys
from typing import Dict, Optional
from rich.console import Console
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

# Model configurations for edge deployment
MODEL_CONFIGS = {
    "qwen3-1.7b-q4": {
        "model_file": "Qwen3-1.7B-Q4_K_M.gguf",
        "model_url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        "model_size_mb": 1110,
        "whisper_file": "ggml-base.bin",
        "whisper_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "whisper_size_mb": 140,
        "description": "Qwen3 1.7B Q4_K_M (1.11GB) 4-bit quantized with Whisper base - Total ~1.25GB",
    },
    "qwen3-1.7b-q4-xs": {
        "model_file": "Qwen3-1.7B-IQ4_XS.gguf",
        "model_url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-IQ4_XS.gguf",
        "model_size_mb": 1010,
        "whisper_file": "ggml-base.bin",
        "whisper_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "whisper_size_mb": 140,
        "description": "Qwen3 1.7B IQ4_XS (1.01GB) smallest 4-bit with Whisper base - Total ~1.15GB",
    },
    "qwen3-1.7b": {
        "model_file": "Qwen3-1.7B-Q8_0.gguf",
        "model_url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
        "model_size_mb": 1830,
        "whisper_file": "ggml-base.bin",
        "whisper_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "whisper_size_mb": 140,
        "description": "Qwen3 1.7B Q8_0 (1.83GB) with advanced reasoning and Whisper base (140MB) - Total ~2GB",
    },
}


class ModelDownloader:
    """Handles model downloading with progress tracking and verification."""

    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_with_progress(
        self, url: str, filepath: Path, expected_size_mb: Optional[int] = None
    ) -> bool:
        """Download file with rich progress bar."""

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                progress.update(task_id, completed=downloaded, total=total_size)

        try:
            with Progress(
                "[progress.description]{task.description}",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                # Use expected size if provided, otherwise use None for indeterminate progress
                total_size = expected_size_mb * 1024 * 1024 if expected_size_mb else None
                task_id = progress.add_task(f"Downloading {filepath.name}", total=total_size)

                urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)

            return True

        except Exception as e:
            console.print(f"❌ [red]Download failed:[/red] {e}")
            return False

    def verify_file_size(
        self, filepath: Path, expected_size_mb: Optional[int] = None, tolerance_mb: int = 100
    ) -> bool:
        """Verify downloaded file size is within expected range."""
        if not filepath.exists():
            return False

        actual_size_mb = filepath.stat().st_size / (1024 * 1024)

        # Skip verification if no expected size provided
        if expected_size_mb is None:
            console.print(f"✅ [green]File downloaded:[/green] {actual_size_mb:.1f} MB")
            return True

        expected_range = (expected_size_mb - tolerance_mb, expected_size_mb + tolerance_mb)

        if expected_range[0] <= actual_size_mb <= expected_range[1]:
            console.print(f"✅ [green]Size verification passed:[/green] {actual_size_mb:.1f} MB")
            return True
        else:
            console.print(
                f"❌ [red]Size mismatch:[/red] Expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB"
            )
            return False

    def download_model_config(self, model_key: str, force_download: bool = False) -> bool:
        """Download complete model configuration."""
        if model_key not in MODEL_CONFIGS:
            console.print(f"❌ [red]Unknown model configuration:[/red] {model_key}")
            return False

        config = MODEL_CONFIGS[model_key]
        model_size_str = f"(~{config['model_size_mb']}MB)" if "model_size_mb" in config else ""
        mmproj_size_str = f"(~{config['mmproj_size_mb']}MB)" if "mmproj_size_mb" in config else ""
        console.print(
            Panel(
                f"[cyan]Model:[/cyan] {config['description']}\n"
                f"[cyan]Model File:[/cyan] {config['model_file']} {model_size_str}\n"
                f"[cyan]Multimodal:[/cyan] {config.get('mmproj_file', 'None')} {mmproj_size_str}",
                title=f"📥 Checking {model_key}",
                border_style="blue",
            )
        )

        success = True

        # Download main model
        model_path = self.models_dir / config["model_file"]
        if force_download or not model_path.exists():
            console.print(f"🔄 [yellow]Downloading model file: {config['model_file']}[/yellow]")
            console.print(f"   URL: {config['model_url']}")
            model_size = config.get("model_size_mb")
            if not self.download_with_progress(config["model_url"], model_path, model_size):
                console.print(f"❌ [red]Failed to download model file[/red]")
                success = False
            elif not self.verify_file_size(model_path, model_size):
                console.print(
                    f"⚠️ [yellow]Model file size verification failed, but keeping file[/yellow]"
                )
        else:
            # Verify existing file size
            actual_size = model_path.stat().st_size / (1024 * 1024)
            console.print(
                f"✅ [green]Model file exists:[/green] {model_path.name} ({actual_size:.1f} MB)"
            )

        # Download Whisper model if needed
        if config.get("whisper_file") and config.get("whisper_url"):
            whisper_path = self.models_dir / config["whisper_file"]
            if force_download or not whisper_path.exists():
                console.print(
                    f"🔄 [yellow]Downloading Whisper model: {config['whisper_file']}[/yellow]"
                )
                console.print(f"   URL: {config['whisper_url']}")
                if not self.download_with_progress(config["whisper_url"], whisper_path):
                    console.print(f"❌ [red]Failed to download Whisper model[/red]")
                    success = False
                else:
                    console.print(f"✅ [green]Whisper model downloaded[/green]")
            else:
                actual_size = whisper_path.stat().st_size / (1024 * 1024)
                console.print(
                    f"✅ [green]Whisper model exists:[/green] {whisper_path.name} ({actual_size:.1f} MB)"
                )

        # Download multimodal projection if needed - INDEPENDENTLY of model
        if config.get("mmproj_file") and config.get("mmproj_url"):
            mmproj_path = self.models_dir / config["mmproj_file"]
            if force_download or not mmproj_path.exists():
                console.print(
                    f"🔄 [yellow]Downloading multimodal projection: {config['mmproj_file']}[/yellow]"
                )
                console.print(f"   URL: {config['mmproj_url']}")
                console.print(
                    f"   ⚠️ Note: This is a large file (~2.4GB) and may take several minutes"
                )
                mmproj_size = config.get("mmproj_size_mb")
                if not self.download_with_progress(config["mmproj_url"], mmproj_path, mmproj_size):
                    console.print(f"❌ [red]Failed to download multimodal projection[/red]")
                    success = False
                elif not self.verify_file_size(mmproj_path, mmproj_size):
                    console.print(
                        f"⚠️ [yellow]Projection file size verification failed, but keeping file[/yellow]"
                    )
            else:
                # Verify existing file size
                actual_size = mmproj_path.stat().st_size / (1024 * 1024)
                console.print(
                    f"✅ [green]Multimodal projection exists:[/green] {mmproj_path.name} ({actual_size:.1f} MB)"
                )

        if success:
            console.print(
                Panel(
                    f"[green]✅ Model configuration ready[/green]\n"
                    f"[cyan]Model:[/cyan] {model_path}\n"
                    f"[cyan]Whisper:[/cyan] {self.models_dir / config.get('whisper_file', 'N/A')}",
                    title="📦 Model Status",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[red]❌ Some files failed to download[/red]\n"
                    f"Please check your internet connection and try again",
                    title="📦 Download Issues",
                    border_style="red",
                )
            )

        return success

    def get_available_space_gb(self) -> float:
        """Get available disk space in GB."""
        import shutil

        total, used, free = shutil.disk_usage(self.models_dir)
        return free / (1024**3)

    def check_space_requirements(self, model_key: str) -> bool:
        """Check if enough disk space is available."""
        if model_key not in MODEL_CONFIGS:
            return False

        available_gb = self.get_available_space_gb()

        # Always show available space and proceed
        console.print(f"💾 [cyan]Available disk space:[/cyan] {available_gb:.1f}GB")

        # Warn if less than 5GB available
        if available_gb < 5:
            console.print(
                f"⚠️ [yellow]Low disk space warning:[/yellow] Models may require several GB"
            )
            return True  # Still proceed, let download fail if needed

        return True


def main():
    """Main function for model download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download models for edge deployment")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument("--model", default=None, help="Target model (e.g., qwen2.5-omni-3b)")
    parser.add_argument("--dir", default=None, help="Models directory path")
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]Personal Assistant Edge Model Downloader[/bold cyan]\n\n"
            "This script downloads the required models for edge deployment of the\n"
            "voice-enabled personal assistant with Qwen2.5-Omni multimodal capabilities.",
            title="🤖 Edge AI Model Downloader",
            border_style="cyan",
        )
    )

    # Determine target model - prioritize CLI arg, then env, then default
    target_model = args.model or os.getenv("TARGET_MODEL", "qwen3-1.7b-q4")

    # Determine models directory - prioritize CLI arg, then env, then check if running locally
    if args.dir:
        models_dir = args.dir
    elif os.getenv("MODELS_DIR"):
        models_dir = os.getenv("MODELS_DIR")
    else:
        # Auto-detect if running locally vs in container
        if os.path.exists("/app/models"):
            models_dir = "/app/models"
        else:
            # Running locally - use the local deployment models directory
            script_dir = Path(__file__).parent
            models_dir = str(script_dir / "deployment" / "models")

    console.print(f"🎯 [cyan]Target Model:[/cyan] {target_model}")
    console.print(f"📁 [cyan]Models Directory:[/cyan] {models_dir}")
    if args.force:
        console.print(f"⚠️ [yellow]Force download:[/yellow] Re-downloading all files")

    downloader = ModelDownloader(models_dir)

    # Check disk space
    if not downloader.check_space_requirements(target_model):
        console.print("❌ [red]Cannot proceed due to insufficient disk space[/red]")
        sys.exit(1)

    # Download the model
    try:
        success = downloader.download_model_config(target_model, force_download=args.force)
        if success:
            console.print("🎉 [green]All models ready![/green]")

            # Set environment variables for the application
            config = MODEL_CONFIGS[target_model]
            model_path = Path(models_dir) / config["model_file"]
            mmproj_path = Path(models_dir) / config.get("mmproj_file", "N/A")

            console.print(
                Panel(
                    f"[green]Ready for deployment![/green]\n\n"
                    f"[cyan]MODEL_PATH:[/cyan] {model_path}\n"
                    f"[cyan]MMPROJ_PATH:[/cyan] {mmproj_path}\n\n"
                    f"[yellow]To use these models:[/yellow]\n"
                    f"[dim]export MODEL_PATH={model_path}[/dim]\n"
                    f"[dim]export MMPROJ_PATH={mmproj_path}[/dim]",
                    title="🚀 Ready for Deployment",
                    border_style="green",
                )
            )

        else:
            console.print("❌ [red]Model download had issues - check files manually[/red]")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Download interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [red]Unexpected error:[/red] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
