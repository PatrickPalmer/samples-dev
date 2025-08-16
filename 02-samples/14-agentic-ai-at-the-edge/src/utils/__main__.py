"""Allow running audio_cli as a module: python -m src.utils.audio_cli"""

from .audio_cli import main
import sys

sys.exit(main())
