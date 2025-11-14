# ============================================================================
# FILE: utils/tool_executor.py
# ============================================================================
"""Tool execution for function calling"""

from typing import Dict, List
from config import Config


class ToolExecutor:
    """Handles tool/function calling capabilities"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.available_tools = {
            "search_transcripts": self.search_transcripts,
            "analyze_sentiment": self.analyze_sentiment,
            "get_statistics": self.get_statistics
        }
    
    def search_transcripts(self, query: str, n_results: int = 3) -> str:
        """Tool: Search conversation transcripts"""
        results = self.vector_db.search(query, n_results)
        
        if not results:
            return "No relevant transcripts found."
        
        output = f"Found {len(results)} relevant transcripts:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['document'][:200]}...\n"
            output += f"   Metadata: {result['metadata']}\n"
            output += f"   Relevance: {1 - result['distance']:.2f}\n\n"
        
        return output
    
    def analyze_sentiment(self, text: str) -> str:
        """Tool: Analyze sentiment"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'satisfied', 'love', 'wonderful']
        negative_words = ['bad', 'poor', 'terrible', 'unhappy', 'disappointed', 'hate', 'awful']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
            score = pos_count / (pos_count + neg_count + 1)
        elif neg_count > pos_count:
            sentiment = "Negative"
            score = neg_count / (pos_count + neg_count + 1)
        else:
            sentiment = "Neutral"
            score = 0.5
        
        return f"Sentiment: {sentiment} (confidence: {score:.2f})"
    
    def get_statistics(self) -> str:
        """Tool: Get database statistics"""
        count = self.vector_db.get_count()
        return f"Total transcripts in database: {count}"
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for LLM"""
        return [
            {
                "name": "search_transcripts",
                "description": "Search through past conversation transcripts",
                "parameters": {
                    "query": "Search query string",
                    "n_results": "Number of results (default: 3)"
                }
            },
            {
                "name": "analyze_sentiment",
                "description": "Analyze sentiment of text",
                "parameters": {
                    "text": "Text to analyze"
                }
            },
            {
                "name": "get_statistics",
                "description": "Get database statistics",
                "parameters": {}
            }
        ]
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> str:
        """Execute a tool with parameters"""
        if tool_name in self.available_tools:
            try:
                return self.available_tools[tool_name](**parameters)
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        return f"Tool {tool_name} not found"

