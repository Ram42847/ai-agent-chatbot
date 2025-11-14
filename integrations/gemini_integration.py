# ============================================================================
# FILE: integrations/gemini_integration.py (ENHANCED VERSION)
# ============================================================================
"""Enhanced Google Gemini Integration with RAG and Tool Calling"""

import os
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from config import Config
from utils.vector_db import VectorDatabase


class GeminiAgent:
    """
    Enhanced Gemini-powered agent with:
    - Native tool calling
    - RAG integration
    - Intent classification
    - Conversation management
    """
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini agent"""
        # Configure API
        self.api_key = api_key or Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Please set it in .env file or pass as parameter."
            )
        
        genai.configure(api_key=self.api_key)
        
        # Initialize vector database (shared with main agent)
        self.vector_db = VectorDatabase()
        
        # Define tools for function calling
        self.tools = self._define_tools()
        
        # Create model with tools
        self.model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            tools=self.tools
        )
        
        # For chat sessions
        self.chat_session = None
        
        print("✓ Gemini Agent initialized with native tool calling")
    
    def _define_tools(self) -> List[Tool]:
        """Define available tools for Gemini function calling"""
        
        # Tool 1: Search Transcripts
        search_transcripts = FunctionDeclaration(
            name="search_transcripts",
            description=(
                "Search through historical conversation transcripts to find "
                "relevant information. Use this when user asks about past "
                "conversations, issues, or patterns."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant transcripts"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        )
        
        # Tool 2: Analyze Sentiment
        analyze_sentiment = FunctionDeclaration(
            name="analyze_sentiment",
            description=(
                "Analyze the sentiment (positive, negative, neutral) of a "
                "given text. Use this when user asks about customer satisfaction "
                "or emotional tone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze for sentiment"
                    }
                },
                "required": ["text"]
            }
        )
        
        # Tool 3: Get Statistics
        get_statistics = FunctionDeclaration(
            name="get_statistics",
            description=(
                "Get statistics about the conversation database including "
                "total count and category breakdown. Use when user asks about "
                "overall metrics or database status."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "include_categories": {
                        "type": "boolean",
                        "description": "Whether to include category breakdown",
                        "default": True
                    }
                }
            }
        )
        
        # Tool 4: Get Recent Conversations
        get_recent_conversations = FunctionDeclaration(
            name="get_recent_conversations",
            description=(
                "Get the most recent conversations from the database. "
                "Use when user asks about recent activity or latest issues."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent conversations to retrieve",
                        "default": 5
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (optional)"
                    }
                }
            }
        )
        
        # Create tool object
        tools = [Tool(
            function_declarations=[
                search_transcripts,
                analyze_sentiment,
                get_statistics,
                get_recent_conversations
            ]
        )]
        
        return tools
    
    def _execute_function(self, function_call) -> str:
        """
        Execute the called function and return results
        
        Args:
            function_call: The function call object from Gemini
            
        Returns:
            String result from the function execution
        """
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        print(f"[Tool] Executing: {function_name}")
        print(f"[Tool] Arguments: {function_args}")
        
        try:
            # Execute based on function name
            if function_name == "search_transcripts":
                return self._tool_search_transcripts(
                    query=function_args.get("query"),
                    limit=function_args.get("limit", 3)
                )
            
            elif function_name == "analyze_sentiment":
                return self._tool_analyze_sentiment(
                    text=function_args.get("text")
                )
            
            elif function_name == "get_statistics":
                return self._tool_get_statistics(
                    include_categories=function_args.get("include_categories", True)
                )
            
            elif function_name == "get_recent_conversations":
                return self._tool_get_recent_conversations(
                    count=function_args.get("count", 5),
                    category=function_args.get("category")
                )
            
            else:
                return f"Unknown function: {function_name}"
        
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def _tool_search_transcripts(self, query: str, limit: int = 3) -> str:
        """Tool implementation: Search transcripts"""
        results = self.vector_db.search(query, n_results=limit)
        
        if not results:
            return "No relevant transcripts found for this query."
        
        output = f"Found {len(results)} relevant transcripts:\n\n"
        for i, result in enumerate(results, 1):
            relevance_score = 1 - result['distance']
            output += f"**Transcript {i}** (Relevance: {relevance_score:.2%})\n"
            output += f"{result['document']}\n"
            output += f"Metadata: {json.dumps(result['metadata'], indent=2)}\n\n"
        
        return output
    
    def _tool_analyze_sentiment(self, text: str) -> str:
        """Tool implementation: Analyze sentiment"""
        # Enhanced sentiment analysis
        positive_words = [
            'good', 'great', 'excellent', 'happy', 'satisfied', 'love',
            'wonderful', 'amazing', 'fantastic', 'pleased', 'delighted'
        ]
        negative_words = [
            'bad', 'poor', 'terrible', 'unhappy', 'disappointed', 'hate',
            'awful', 'horrible', 'frustrated', 'angry', 'upset'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            sentiment = "Neutral"
            confidence = 0.5
        elif pos_count > neg_count:
            sentiment = "Positive"
            confidence = pos_count / total
        else:
            sentiment = "Negative"
            confidence = neg_count / total
        
        return (
            f"**Sentiment Analysis Results:**\n"
            f"- Overall Sentiment: {sentiment}\n"
            f"- Confidence: {confidence:.2%}\n"
            f"- Positive indicators: {pos_count}\n"
            f"- Negative indicators: {neg_count}\n"
            f"- Analysis: The text shows {sentiment.lower()} sentiment "
            f"with {confidence:.0%} confidence based on keyword analysis."
        )
    
    def _tool_get_statistics(self, include_categories: bool = True) -> str:
        """Tool implementation: Get database statistics"""
        count = self.vector_db.get_count()
        
        output = f"**Database Statistics:**\n"
        output += f"- Total transcripts: {count}\n"
        
        if include_categories and count > 0:
            # Get all documents with metadata
            try:
                all_docs = self.vector_db.collection.get()
                categories = {}
                
                for metadata in all_docs['metadatas']:
                    cat = metadata.get('category', 'uncategorized')
                    categories[cat] = categories.get(cat, 0) + 1
                
                output += f"\n**Category Breakdown:**\n"
                for cat, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    percentage = (cnt / count) * 100
                    output += f"- {cat}: {cnt} ({percentage:.1f}%)\n"
            except Exception as e:
                output += f"\n(Could not retrieve category breakdown: {e})"
        
        return output
    
    def _tool_get_recent_conversations(self, count: int = 5, category: str = None) -> str:
        """Tool implementation: Get recent conversations"""
        # This is a simplified implementation
        # In production, you'd want to maintain timestamps and query by date
        try:
            all_docs = self.vector_db.collection.get()
            
            if not all_docs['documents']:
                return "No conversations found in database."
            
            # Filter by category if specified
            filtered_docs = []
            for i, doc in enumerate(all_docs['documents']):
                metadata = all_docs['metadatas'][i]
                if category and metadata.get('category') != category:
                    continue
                filtered_docs.append({
                    'document': doc,
                    'metadata': metadata
                })
            
            # Get most recent (last N)
            recent = filtered_docs[-count:] if len(filtered_docs) > count else filtered_docs
            
            output = f"**Recent Conversations** ({len(recent)} found):\n\n"
            for i, item in enumerate(reversed(recent), 1):
                output += f"**{i}.** {item['document'][:150]}...\n"
                output += f"   Category: {item['metadata'].get('category', 'N/A')}\n"
                output += f"   Date: {item['metadata'].get('date', 'N/A')}\n\n"
            
            return output
        
        except Exception as e:
            return f"Error retrieving recent conversations: {str(e)}"
    
    def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Classify user intent using Gemini
        
        Args:
            user_input: User's message
            
        Returns:
            Dict with intent, confidence, and reasoning
        """
        prompt = f"""Classify the intent of this user message.

User Message: "{user_input}"

Available Intent Categories:
- customer_query: General questions about products/services
- complaint: User expressing dissatisfaction or problems
- feedback: User providing positive or constructive feedback
- support_request: Technical support or help needed
- product_inquiry: Questions about specific products
- billing_issue: Payment, invoice, or billing related
- general_conversation: Casual conversation or greetings

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "intent": "category_name",
  "confidence": 0.95,
  "reasoning": "brief explanation"
}}"""

        try:
            response = genai.GenerativeModel('gemini-pro').generate_content(
                prompt,
                generation_config={'temperature': 0.1}
            )
            
            # Clean and parse response
            text = response.text.strip()
            # Remove markdown code blocks if present
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(text)
            return result
        
        except Exception as e:
            print(f"[Error] Intent classification failed: {e}")
            return {
                "intent": "general_conversation",
                "confidence": 0.5,
                "reasoning": "Could not classify intent"
            }
    
    def chat_with_rag(
        self, 
        user_query: str, 
        context: List[str] = None,
        intent: str = None
    ) -> Dict[str, Any]:
        """
        Process query with RAG and automatic tool calling
        
        Args:
            user_query: User's question
            context: Pre-retrieved context documents (optional)
            intent: Pre-classified intent (optional)
            
        Returns:
            Dict with response and function calls made
        """
        # Build prompt with context
        if context:
            context_text = "\n\n".join([
                f"Context {i+1}: {doc}"
                for i, doc in enumerate(context)
            ])
            
            prompt = f"""You are a helpful AI assistant with access to historical conversation data and specialized tools.

**Context from past conversations:**
{context_text}

**User's question:** {user_query}
{f"**Detected intent:** {intent}" if intent else ""}

**Instructions:**
- Use the provided context to inform your response
- If you need additional information, use the available tools
- Be conversational and helpful
- Cite specific context when relevant

Provide a helpful response:"""
        else:
            prompt = user_query
        
        # Initialize chat session if needed
        if not self.chat_session:
            self.chat_session = self.model.start_chat()
        
        # Send message
        response = self.chat_session.send_message(prompt)
        
        # Track function calls
        function_calls_made = []
        
        # Handle function calling loop
        while response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            # Check if function was called
            if hasattr(part, 'function_call'):
                function_call = part.function_call
                
                # Execute function
                function_response = self._execute_function(function_call)
                
                # Track call
                function_calls_made.append({
                    'name': function_call.name,
                    'args': dict(function_call.args),
                    'response': function_response
                })
                
                # Send function response back to model
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
                # No more function calls, we have final response
                break
        
        # Extract final text response
        final_response = response.text if hasattr(response, 'text') else str(response)
        
        return {
            'response': final_response,
            'function_calls': function_calls_made,
            'raw_response': response
        }
    
    def simple_query(self, query: str) -> str:
        """
        Simple one-off query without chat history
        
        Args:
            query: User's question
            
        Returns:
            String response
        """
        response = self.model.generate_content(query)
        return response.text
    
    def reset_chat(self):
        """Reset chat session"""
        self.chat_session = None
        print("Chat session reset")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Example usage of GeminiAgent"""
    
    # Initialize agent
    agent = GeminiAgent()
    
    # Example 1: Simple intent classification
    print("\n=== Example 1: Intent Classification ===")
    intent = agent.classify_intent("My bill seems incorrect this month")
    print(f"Intent: {intent['intent']}")
    print(f"Confidence: {intent['confidence']}")
    print(f"Reasoning: {intent['reasoning']}")
    
    # Example 2: Query with automatic tool calling
    print("\n=== Example 2: Query with Tool Calling ===")
    result = agent.chat_with_rag(
        user_query="What billing issues have we had recently?",
        intent="billing_issue"
    )
    print(f"Response: {result['response']}")
    print(f"Tools used: {len(result['function_calls'])}")
    
    # Example 3: Query with pre-retrieved context
    print("\n=== Example 3: Query with RAG Context ===")
    context = [
        "Customer reported billing discrepancy. Refund processed.",
        "Technical support for login problems. Issue resolved."
    ]
    result = agent.chat_with_rag(
        user_query="Tell me about recent customer issues",
        context=context
    )
    print(f"Response: {result['response']}")


if __name__ == "__main__":
    example_usage()