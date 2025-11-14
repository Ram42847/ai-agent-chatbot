# ============================================================================
# FILE: utils/intent_classifier.py
# ============================================================================
"""Intent classification using LLM"""

import json
from typing import Tuple
import ollama
from config import Config
from prompts.intent_prompts import INTENT_CLASSIFICATION_PROMPT


class IntentClassifier:
    """Classifies user intent using LLM"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.OLLAMA_MODEL
        self.intent_categories = Config.INTENT_CATEGORIES
        
    def classify(self, user_input: str) -> Tuple[str, float]:
        """
        Classify the intent of user input
        
        Returns:
            Tuple of (intent, confidence)
        """
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_input=user_input,
            categories=', '.join(self.intent_categories)
        )
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            response_text = response['response'].strip()
            
            # Extract JSON from response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            result = json.loads(response_text)
            intent = result.get('intent', 'general_conversation')
            confidence = result.get('confidence', 0.5)
            
            # Validate intent
            if intent not in self.intent_categories:
                intent = 'general_conversation'
            
            return intent, confidence
            
        except Exception as e:
            print(f"Intent classification error: {e}")
            return "general_conversation", 0.5

