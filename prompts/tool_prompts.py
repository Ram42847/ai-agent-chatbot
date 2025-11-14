# ============================================================================
# FILE: prompts/tool_prompts.py
# ============================================================================
"""Prompts for tool calling"""

TOOL_CALLING_PROMPT = """You are an AI assistant with access to specialized tools.

USER REQUEST: {user_input}

AVAILABLE TOOLS:
{tool_definitions}

Decision Process:
1. Analyze if the request requires tool usage
2. Determine which tool(s) would be most helpful
3. Extract necessary parameters from user input

If a tool is needed, respond with JSON:
{{
  "use_tool": true,
  "tool_name": "tool_name",
  "parameters": {{"param1": "value1"}}
}}

If no tool is needed, respond with JSON:
{{
  "use_tool": false,
  "direct_response": "your answer here"
}}

Analysis:"""


print("All file structures defined successfully!")
print("\nNext steps:")
print("1. Create the directory structure")
print("2. Copy each section into its respective file")
print("3. Run setup.sh to initialize the project")