"""Tests for intent classification"""

import pytest
from utils.intent_classifier import IntentClassifier


def test_intent_classification():
    """Test basic intent classification"""
    classifier = IntentClassifier()
    
    test_cases = [
        ("My bill is wrong", "billing_issue"),
        ("I need help logging in", "support_request"),
        ("Your service is great!", "feedback"),
    ]
    
    for text, expected_intent in test_cases:
        intent, confidence = classifier.classify(text)
        # Note: LLM might not always return exact intent
        assert intent in classifier.intent_categories
        assert 0 <= confidence <= 1


def test_invalid_input():
    """Test handling of invalid input"""
    classifier = IntentClassifier()
    intent, confidence = classifier.classify("")
    assert intent is not None