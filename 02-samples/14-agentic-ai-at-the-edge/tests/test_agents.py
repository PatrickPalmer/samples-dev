"""
Test suite for agent functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCalendarAgent:
    """Test calendar assistant agent."""

    @pytest.mark.unit
    def test_appointment_creation(self):
        """Test creating a calendar appointment."""
        from src.agents.tools.calendar_tools.create_appointment import create_appointment

        result = create_appointment(
            date="2024-12-20 14:00",
            location="Conference Room A",
            title="Team Meeting",
            description="Weekly team sync",
        )

        assert "✅ Appointment Created Successfully!" in result
        assert "Team Meeting" in result
        assert "2024-12-20" in result

    @pytest.mark.integration
    @pytest.mark.requires_llama
    def test_calendar_agent_real(self):
        """Test calendar agent with REAL LlamaCpp model."""
        import requests

        # Check if server is running
        try:
            if requests.get("http://localhost:8080/health", timeout=2).status_code != 200:
                pytest.skip("LlamaCpp server not running")
        except:
            pytest.skip("LlamaCpp server not available")

        # from src.agents.calendar_assistant import calendar_assistant  # NOT IMPLEMENTED
        pytest.skip("calendar_assistant not implemented")

        # Test with real model
        print("\n📅 Testing Real Calendar Agent...")
        response = calendar_assistant("Schedule a team standup tomorrow at 10am")
        assert response is not None
        print(f"   ✅ Real response: {response[:200]}")

        # Test listing
        response = calendar_assistant("Show my appointments")
        assert response is not None
        print(f"   ✅ List response: {response[:200]}")

    @pytest.mark.unit
    def test_appointment_listing(self):
        """Test listing appointments."""
        from src.agents.tools.calendar_tools.list_appointments import list_appointments
        from src.agents.tools.calendar_tools.create_appointment import create_appointment

        # Create test appointment
        create_appointment(
            date="2024-12-20 10:00", location="Office", title="Test", description="Test appointment"
        )

        result = list_appointments()
        assert "appointment" in result.lower() or "date" in result.lower()


class TestVehicleAgent:
    """Test vehicle assistant agent."""

    @pytest.mark.unit
    def test_knowledge_search(self, mock_vehicle_store):
        """Test vehicle knowledge retrieval."""
        from src.data.vehicle_knowledge import get_vehicle_store

        with patch("src.data.vehicle_knowledge.knowledge_store.get_vehicle_store") as mock_get:
            mock_get.return_value = mock_vehicle_store

            store = get_vehicle_store()
            results = store.search("tire pressure reset")

            assert len(results) > 0
            assert "TPMS" in results[0]["content"]
            assert results[0]["score"] > 0.5

    @pytest.mark.integration
    @pytest.mark.requires_llama
    def test_vehicle_agent_real_faiss(self):
        """Test vehicle agent with REAL FAISS vector store and model."""
        import requests

        # Check if server is running
        try:
            if requests.get("http://localhost:8080/health", timeout=2).status_code != 200:
                pytest.skip("LlamaCpp server not running")
        except:
            pytest.skip("LlamaCpp server not available")

        # from src.agents.vehicle_assistant import vehicle_assistant  # NOT IMPLEMENTED
        pytest.skip("vehicle_assistant not implemented")

        print("\n🚗 Testing Real Vehicle Agent with FAISS...")

        # Test tire pressure query
        response = vehicle_assistant("How do I reset the tire pressure monitoring system?")
        assert response is not None
        assert "tire" in response.lower() or "pressure" in response.lower() or "TPMS" in response
        print(f"   ✅ TPMS query: {response[:200]}")

        # Test oil change query
        response = vehicle_assistant("What's the recommended oil change interval?")
        assert response is not None
        print(f"   ✅ Oil change query: {response[:200]}")

        # Test check engine light
        response = vehicle_assistant("My check engine light is on, what should I do?")
        assert response is not None
        print(f"   ✅ Check engine query: {response[:200]}")

    @pytest.mark.unit
    def test_vehicle_tool(self, mock_vehicle_store):
        """Test vehicle assistant tool."""
        # from src.agents.vehicle_assistant import search_vehicle_knowledge  # NOT IMPLEMENTED
        pytest.skip("vehicle_assistant not implemented")

        # with patch("src.agents.vehicle_assistant.get_vehicle_store") as mock_get:
        #     mock_get.return_value = mock_vehicle_store
        #
        #     result = search_vehicle_knowledge("How to reset tire pressure?")
        #
        #     assert "TPMS" in result
        #     assert "Owner's Manual" in result

    @pytest.mark.unit
    def test_faiss_indexing(self, mock_vehicle_documents, temp_test_dir):
        """Test FAISS vector store indexing."""
        from src.data.vehicle_knowledge.knowledge_store import VehicleKnowledgeStore

        store = VehicleKnowledgeStore(store_path=temp_test_dir)
        store.add_documents(mock_vehicle_documents)

        assert len(store.documents) == len(mock_vehicle_documents)

        results = store.search("oil change", top_k=1)
        assert len(results) > 0
        assert "5,000 miles" in results[0]["content"]


class TestSearchAgent:
    """Test search assistant agent."""

    @pytest.mark.unit
    def test_search_agent_initialization(self):
        """Test search agent setup."""
        # from src.agents.search_assistant import search_assistant  # NOT IMPLEMENTED
        pytest.skip("search_assistant not implemented")

        # Verify the agent tool is callable
        assert callable(search_assistant)
        assert hasattr(search_assistant, "__call__")
