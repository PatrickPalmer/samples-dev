"""
Mock data and utilities for testing.
"""

import numpy as np
from typing import Dict, Any, List


class MockAudioGenerator:
    """Generate mock audio data for testing."""

    @staticmethod
    def generate_audio(duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
        """Generate mock audio as numpy array."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        return audio.astype(np.float32)

    @staticmethod
    def to_bytes(audio: np.ndarray) -> bytes:
        """Convert audio array to bytes."""
        return audio.tobytes()


class MockTranscriptionService:
    """Mock voice transcription scenarios."""

    SCENARIOS = {
        "calendar": [
            ("schedule a meeting tomorrow at 2pm", "calendar"),
            ("what's on my calendar today", "calendar"),
            ("cancel my dentist appointment", "calendar"),
        ],
        "vehicle": [
            ("how do I reset the tire pressure monitor", "vehicle"),
            ("what's the oil change interval", "vehicle"),
            ("my check engine light is on", "vehicle"),
        ],
        "search": [
            ("what's the weather today", "search"),
            ("find news about AI", "search"),
            ("search for restaurants nearby", "search"),
        ],
        "general": [
            ("what time is it", "simple"),
            ("hello how are you", "simple"),
            ("tell me a joke", "simple"),
        ],
    }

    @classmethod
    def get_scenario(cls, category: str = "general", index: int = 0) -> Dict[str, Any]:
        """Get a mock transcription scenario."""
        scenarios = cls.SCENARIOS.get(category, cls.SCENARIOS["general"])
        text, task_type = scenarios[index % len(scenarios)]

        return {
            "status": "success",
            "user_request": text,
            "raw_transcription": f"User said: {text}",
            "category": category,
            "task_type": task_type,
        }

    @classmethod
    def random_scenario(cls) -> Dict[str, Any]:
        """Get a random scenario."""
        import random

        category = random.choice(list(cls.SCENARIOS.keys()))
        index = random.randint(0, len(cls.SCENARIOS[category]) - 1)
        return cls.get_scenario(category, index)


class MockVehicleKnowledge:
    """Mock vehicle knowledge base."""

    KNOWLEDGE = {
        "tire pressure": {
            "content": "To reset TPMS: 1. Turn ignition ON 2. Hold TPMS button 3 seconds",
            "source": "Owner's Manual",
            "page": 142,
        },
        "oil change": {
            "content": "Change oil every 5,000 miles or 6 months",
            "source": "Maintenance Guide",
            "page": 98,
        },
        "check engine": {
            "content": "Check engine light may indicate: loose gas cap, O2 sensor issue",
            "source": "Troubleshooting",
            "page": 205,
        },
    }

    @classmethod
    def search(cls, query: str) -> List[Dict[str, Any]]:
        """Search mock knowledge base."""
        query_lower = query.lower()
        results = []

        for key, info in cls.KNOWLEDGE.items():
            if key in query_lower:
                results.append(
                    {
                        "content": info["content"],
                        "metadata": {"source": info["source"], "page": info["page"]},
                        "score": 0.85,
                    }
                )

        return (
            results
            if results
            else [
                {
                    "content": f"No specific information found for: {query}",
                    "metadata": {"source": "General"},
                    "score": 0.3,
                }
            ]
        )
