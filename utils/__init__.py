# ============================================================================
# FILE: prompts/__init__.py
# ============================================================================
"""Prompt templates for AI Agent"""

from .intent_prompts import INTENT_CLASSIFICATION_PROMPT
from .rag_prompts import RAG_RESPONSE_PROMPT, SYSTEM_PROMPT
from .tool_prompts import TOOL_CALLING_PROMPT

__all__ = [
    'INTENT_CLASSIFICATION_PROMPT',
    'RAG_RESPONSE_PROMPT',
    'SYSTEM_PROMPT',
    'TOOL_CALLING_PROMPT'
]