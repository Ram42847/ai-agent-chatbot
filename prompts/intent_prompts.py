# ============================================================================
# FILE: prompts/intent_prompts.py
# ============================================================================
"""Prompts for intent classification"""

INTENT_CLASSIFICATION_PROMPT = """You are an expert intent classifier for customer service conversations.

Analyze the following user message and determine the primary intent.

User Message: "{user_input}"

Available Intent Categories:
{categories}

Classification Guidelines:
- Focus on the primary intent, not secondary themes
- Consider emotional tone and urgency
- Look for keywords and context clues

Respond ONLY with valid JSON in this exact format:
{{
  "intent": "category_name",
  "confidence": 0.95
}}

JSON Response:"""