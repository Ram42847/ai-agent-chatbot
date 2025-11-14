"""Tests for main AI agent"""

import pytest
from main import AIAgent


@pytest.fixture
def agent():
    """Create agent fixture"""
    return AIAgent()


def test_agent_initialization(agent):
    """Test agent initializes correctly"""
    assert agent is not None
    assert agent.vector_db is not None
    assert agent.intent_classifier is not None


def test_process_query(agent):
    """Test query processing"""
    result = agent.process_query("Hello")
    assert 'response' in result
    assert 'intent' in result
    assert 'confidence' in result


def test_conversation_history(agent):
    """Test conversation history tracking"""
    agent.process_query("First message")
    agent.process_query("Second message")
    assert len(agent.conversation_history) == 2