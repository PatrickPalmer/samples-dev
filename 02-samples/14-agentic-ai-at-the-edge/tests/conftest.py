"""
Shared pytest fixtures and configuration.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ==================== Mock Fixtures ====================


@pytest.fixture
def mock_vehicle_store():
    """Mock vehicle knowledge store."""
    store = MagicMock()
    store.search.return_value = [
        {
            "content": "Reset tire pressure by holding the TPMS button for 3 seconds",
            "metadata": {"section": "Maintenance", "page": 142},
            "source": "Owner's Manual",
            "score": 0.85,
        }
    ]
    return store


@pytest.fixture
def mock_audio_recorder():
    """Mock audio recorder."""
    recorder = MagicMock()
    # Generate fake audio data (5 seconds at 16kHz)
    fake_audio = np.random.randn(5 * 16000, 1).astype(np.float32)
    recorder.record.return_value = fake_audio
    recorder.save_to_bytes.return_value = b"fake_audio_bytes"
    return recorder


@pytest.fixture
def mock_voice_input():
    """Mock voice input tool."""

    def voice_input_side_effect(duration=5):
        return {
            "status": "success",
            "content": [{"text": "Voice input captured"}],
            "user_request": "schedule a meeting tomorrow at 2pm",
            "raw_transcription": "schedule a meeting tomorrow at 2pm",
            "audio_duration": duration,
            "model_info": {"supported": True, "model": "qwen2.5-omni"},
        }

    mock = MagicMock()
    mock.side_effect = voice_input_side_effect
    return mock


# ==================== Test Data Fixtures ====================


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "simple": ["What time is it?", "Hello", "Show my calendar"],
        "complex": [
            "Analyze the quarterly revenue trends",
            "Write a detailed marketing strategy",
            "Explain quantum computing applications",
        ],
        "calendar": ["Schedule a meeting tomorrow at 2pm", "What's on my calendar today?"],
        "vehicle": ["How do I reset the tire pressure monitor?", "What's the oil change interval?"],
    }


@pytest.fixture
def mock_vehicle_documents():
    """Mock vehicle documentation."""
    return [
        {
            "content": "To reset the TPMS, turn ignition ON and hold TPMS button for 3 seconds",
            "metadata": {"section": "Maintenance", "page": 142},
            "source": "Owner's Manual",
        },
        {
            "content": "Engine oil should be changed every 5,000 miles or 6 months",
            "metadata": {"section": "Maintenance", "page": 98},
            "source": "Maintenance Guide",
        },
    ]


@pytest.fixture
def temp_test_dir(tmp_path):
    """Temporary directory for tests."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


# ==================== Performance Fixtures ====================


@pytest.fixture
def benchmark():
    """Simple benchmark fixture."""
    import time

    class Benchmark:
        def __init__(self):
            self.times = []

        def __call__(self, func, *args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            self.times.append(time.time() - start)
            return result

        @property
        def avg(self):
            return sum(self.times) / len(self.times) if self.times else 0

    return Benchmark()


# ==================== Pytest Configuration ====================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_llama: Requires llama.cpp server")
    config.addinivalue_line("markers", "requires_bedrock: Requires AWS Bedrock")
