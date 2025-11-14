# ============================================================================
# FILE: tests/test_vector_db.py
# ============================================================================
"""Tests for vector database"""

import pytest
from utils.vector_db import VectorDatabase
from config import Config


def test_vector_db_initialization():
    """Test database initialization"""
    db = VectorDatabase()
    assert db is not None
    assert db.collection is not None


def test_add_transcript():
    """Test adding a transcript"""
    db = VectorDatabase()
    doc_id = db.add_transcript(
        "Test transcript",
        {"date": "2024-01-01", "category": "test"}
    )
    assert doc_id is not None


def test_search():
    """Test searching transcripts"""
    db = VectorDatabase()
    db.add_transcript("Billing issue", {"category": "billing"})
    results = db.search("billing problem")
    assert len(results) > 0


print("Test file created. Run with: pytest tests/")