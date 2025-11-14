"""Google Gemini Integration with Tool Calling"""

import os
import json
from typing import Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from config import Config
from utils.vector_db import VectorDatabase


class GeminiAgent:
    """Gemini-powered agent with native tool calling"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in config")
        
        genai.configure(api_key=self.api_key)
        self.vector_db = VectorDatabase()
        self.tools = self._define_tools()
        self.model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            tools=self.tools
        )
        self.chat_session = None
    
    def _define_tools(self) -> List[Tool]:
        """Define available tools for Gemini"""
        search_transcripts = FunctionDeclaration(
            name="search_transcripts",
            description="Search through historical conversation transcripts",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        )
        
        analyze_sentiment = FunctionDeclaration(
            name="analyze_sentiment",
            description="Analyze sentiment of text",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            }
        )
        
        return [Tool(function_declarations=[search_transcripts, analyze_sentiment])]
    
    def _execute_function(self, function_call) -> str:
        """Execute called function"""
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        if function_name == "search_transcripts":
            query = function_args.get("query")
            limit = function_args.get("limit", 3)
            results = self.vector_db.search(query, n_results=limit)
            
            output = f"Found {len(results)} relevant transcripts:\n\n"
            for i, result in enumerate(results, 1):
                output += f"{i}. {result['document']}\n"
            return output
        
        elif function_name == "analyze_sentiment":
            text = function_args.get("text")
            # Simple sentiment logic
            positive = sum(1 for w in ['good', 'great', 'excellent'] if w in text.lower())
            negative = sum(1 for w in ['bad', 'poor', 'terrible'] if w in text.lower())
            
            if positive > negative:
                return "Sentiment: Positive"
            elif negative > positive:
                return "Sentiment: Negative"
            return "Sentiment: Neutral"
        
        return f"Unknown function: {function_name}"
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process message with automatic tool calling"""
        if not self.chat_session:
            self.chat_session = self.model.start_chat()
        
        response = self.chat_session.send_message(user_message)
        function_calls_made = []
        
        while response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if hasattr(part, 'function_call'):
                function_call = part.function_call
                function_response = self._execute_function(function_call)
                
                function_calls_made.append({
                    'name': function_call.name,
                    'args': dict(function_call.args),
                    'response': function_response
                })
                
                response = self.chat_session.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_call.name,
                                response={'result': function_response}
                            )
                        )]
                    )
                )
            else:
                break
        
        return {
            'response': response.text,
            'function_calls': function_calls_made
        }