# ============================================================================
# FILE: main.py
# ============================================================================
"""Main application file for AI Agent Chatbot"""

import json
import time
from typing import Dict, Any
from datetime import datetime

import gradio as gr
import ollama

from config import Config
from utils.vector_db import VectorDatabase
from utils.intent_classifier import IntentClassifier
from utils.voice_processor import VoiceTranscriber
from utils.tool_executor import ToolExecutor
from prompts.rag_prompts import RAG_RESPONSE_PROMPT, SYSTEM_PROMPT
from prompts.tool_prompts import TOOL_CALLING_PROMPT


class AIAgent:
    """Main AI Agent orchestrating all components"""
    
    def __init__(self, model_name: str = None):
        print("Initializing AI Agent...")
        
        # Ensure directories exist
        Config.ensure_directories()
        
        # Initialize components
        self.model_name = model_name or Config.OLLAMA_MODEL
        self.vector_db = VectorDatabase()
        self.intent_classifier = IntentClassifier(self.model_name)
        self.voice_transcriber = VoiceTranscriber()
        self.tool_executor = ToolExecutor(self.vector_db)
        self.conversation_history = []
        
        print("✓ AI Agent initialized successfully")
        
    def process_query(self, user_input: str, use_rag: bool = True) -> Dict[str, Any]:
        """Process user query through the complete pipeline"""
        
        start_time = time.time()
        
        # Step 1: Intent Classification
        print(f"\n[Processing] User query: {user_input[:50]}...")
        intent, confidence = self.intent_classifier.classify(user_input)
        print(f"[Intent] {intent} (confidence: {confidence:.2f})")
        
        # Step 2: RAG - Retrieve relevant context
        relevant_context = ""
        if use_rag:
            results = self.vector_db.search(user_input, n_results=Config.TOP_K_RESULTS)
            if results:
                relevant_context = "\n\n".join([
                    f"Transcript {i+1}: {r['document']}"
                    for i, r in enumerate(results)
                ])
                print(f"[RAG] Retrieved {len(results)} relevant documents")
        
        # Step 3: Build prompt with context
        system_prompt = SYSTEM_PROMPT.format(
            tool_definitions=json.dumps(
                self.tool_executor.get_tool_definitions(), 
                indent=2
            )
        )
        
        user_prompt = RAG_RESPONSE_PROMPT.format(
            retrieved_context=relevant_context if relevant_context else 'No relevant past conversations found.',
            user_query=user_input,
            intent=intent
        )
        
        # Step 4: Generate response with LLM
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=f"{system_prompt}\n\n{user_prompt}",
                options={
                    "temperature": Config.OLLAMA_TEMPERATURE,
                    "top_p": Config.OLLAMA_TOP_P
                }
            )
            
            response_text = response['response']
            
            # Step 5: Check for tool calling
            tool_response = None
            if '"use_tool": true' in response_text or '"use_tool":true' in response_text:
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        tool_call = json.loads(response_text[json_start:json_end])
                        
                        if tool_call.get('use_tool'):
                            tool_name = tool_call.get('tool_name')
                            parameters = tool_call.get('parameters', {})
                            print(f"[Tool] Executing {tool_name}")
                            tool_response = self.tool_executor.execute_tool(tool_name, parameters)
                            
                            # Generate final response with tool result
                            final_prompt = f"""User asked: {user_input}

Tool {tool_name} returned:
{tool_response}

Provide a natural response incorporating this information."""
                            
                            final_response = ollama.generate(
                                model=self.model_name,
                                prompt=final_prompt,
                                options={"temperature": Config.OLLAMA_TEMPERATURE}
                            )
                            response_text = final_response['response']
                except Exception as e:
                    print(f"[Error] Tool calling error: {e}")
            
            processing_time = time.time() - start_time
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_input,
                "assistant": response_text,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only recent history
            if len(self.conversation_history) > Config.MAX_CONVERSATION_HISTORY:
                self.conversation_history = self.conversation_history[-Config.MAX_CONVERSATION_HISTORY:]
            
            print(f"[Complete] Processing time: {processing_time:.2f}s")
            
            return {
                "response": response_text,
                "intent": intent,
                "confidence": confidence,
                "tool_used": tool_response is not None,
                "tool_response": tool_response,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"[Error] {str(e)}")
            return {
                "response": f"I encountered an error processing your request. Please try again.",
                "intent": intent,
                "confidence": confidence,
                "tool_used": False,
                "tool_response": None,
                "processing_time": time.time() - start_time
            }
    
    def process_voice_query(self, audio_file) -> Dict[str, Any]:
        """Process voice input"""
        try:
            print("[Voice] Transcribing audio...")
            transcription = self.voice_transcriber.transcribe_audio(audio_file)
            print(f"[Voice] Transcribed: {transcription}")
            
            if not transcription:
                return {"response": "Could not transcribe audio. Please try again."}
            
            return self.process_query(transcription)
        except Exception as e:
            print(f"[Error] Voice processing error: {e}")
            return {"response": f"Error processing voice input: {str(e)}"}
    
    def load_sample_data(self):
        """Load sample conversation transcripts"""
        sample_file = f"{Config.DATA_DIR}/sample_transcripts.json"
        
        try:
            with open(sample_file, 'r') as f:
                transcripts = json.load(f)
            self.vector_db.bulk_add_transcripts(transcripts)
            print(f"✓ Loaded {len(transcripts)} sample transcripts")
        except FileNotFoundError:
            print("⚠ Sample data file not found. Creating default samples...")
            self._create_default_samples()
    
    def _create_default_samples(self):
        """Create default sample data"""
        sample_transcripts = [
            {
                "text": "Customer called about billing issue. Account number 12345. Duplicate charge identified. Refund processed within 24 hours. Customer satisfied with resolution.",
                "metadata": {"date": "2024-01-15", "agent": "John Smith", "category": "billing", "sentiment": "resolved"}
            },
            {
                "text": "Technical support call for login problems. User unable to access dashboard after password reset. Cleared browser cache and cookies. Issue resolved. Provided security tips.",
                "metadata": {"date": "2024-01-16", "agent": "Sarah Johnson", "category": "technical", "sentiment": "satisfied"}
            },
            {
                "text": "Sales inquiry about enterprise plan features. Customer managing 150+ users. Discussed custom pricing, dedicated support, and API access. Sent detailed proposal.",
                "metadata": {"date": "2024-01-17", "agent": "Mike Chen", "category": "sales", "sentiment": "interested"}
            },
            {
                "text": "Customer complaint about delayed delivery. Order #9876. Package tracking showed warehouse delay. Expedited shipping arranged at no cost. Compensation provided.",
                "metadata": {"date": "2024-01-18", "agent": "Lisa Wong", "category": "complaint", "sentiment": "resolved"}
            },
            {
                "text": "Positive feedback call. Customer very satisfied with new dashboard features. Praised improved user interface and faster load times. Requested additional features.",
                "metadata": {"date": "2024-01-19", "agent": "David Lee", "category": "feedback", "sentiment": "positive"}
            }
        ]
        
        self.vector_db.bulk_add_transcripts(sample_transcripts)
        
        # Save to file
        with open(f"{Config.DATA_DIR}/sample_transcripts.json", 'w') as f:
            json.dump(sample_transcripts, f, indent=2)


def create_gradio_interface(agent: AIAgent):
    """Create Gradio interface for the chatbot"""
    
    def chat_response(message, history):
        """Handle chat messages"""
        result = agent.process_query(message)
        response = result['response']
        
        # Add metadata
        metadata = f"\n\n---\n*Intent: {result['intent']} | "
        metadata += f"Confidence: {result['confidence']:.2f} | "
        metadata += f"Time: {result['processing_time']:.2f}s*"
        
        if result['tool_used']:
            metadata += f"\n*Tool used to retrieve information*"
        
        return response + metadata
    
    def voice_response(audio):
        """Handle voice input"""
        if audio is None:
            return "Please record audio first."
        
        result = agent.process_voice_query(audio)
        return result['response']
    
    def add_transcript(text, metadata_str):
        """Add new transcript to database"""
        try:
            metadata = json.loads(metadata_str)
            doc_id = agent.vector_db.add_transcript(text, metadata)
            return f"✓ Transcript added successfully! (ID: {doc_id})"
        except Exception as e:
            return f"✗ Error: {str(e)}"
    
    def get_stats():
        """Get database statistics"""
        count = agent.vector_db.get_count()
        return f"Total transcripts in database: {count}"
    
    # Create interface
    with gr.Blocks(
        title="AI Agent Chatbot", 
        theme=gr.themes.Soft(),
        css="""
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="header">
            <h1>🤖 AI Agent Chatbot with RAG</h1>
            <p>Powered by Ollama, ChromaDB, and Whisper</p>
        </div>
        """)
        
        with gr.Tab("💬 Text Chat"):
            gr.Markdown("""
            Ask questions and the agent will search past conversations to provide contextual responses.
            The agent can also use tools to search, analyze sentiment, and get statistics.
            """)
            chatbot = gr.Chatbot(height=500, show_label=False)
            msg = gr.Textbox(
                label="Your Message", 
                placeholder="Type your question here...",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            
            msg.submit(chat_response, [msg, chatbot], [chatbot])
            submit.click(chat_response, [msg, chatbot], [chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Tab("🎤 Voice Chat"):
            gr.Markdown("Record your voice and get a response from the AI agent.")
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="Record Your Voice"
            )
            voice_output = gr.Textbox(label="Response", lines=10)
            voice_button = gr.Button("Process Voice", variant="primary")
            
            voice_button.click(voice_response, inputs=audio_input, outputs=voice_output)
        
        with gr.Tab("📊 Database Management"):
            gr.Markdown("### Add New Transcript")
            
            with gr.Row():
                with gr.Column():
                    new_text = gr.Textbox(
                        label="Transcript Text", 
                        lines=5,
                        placeholder="Enter conversation transcript..."
                    )
                    new_metadata = gr.Textbox(
                        label="Metadata (JSON)", 
                        lines=3,
                        value='{"date": "2024-01-20", "agent": "Agent Name", "category": "general", "sentiment": "neutral"}'
                    )
                    add_button = gr.Button("Add Transcript", variant="primary")
                    add_output = gr.Textbox(label="Status", lines=2)
                
                with gr.Column():
                    gr.Markdown("### Database Statistics")
                    stats_button = gr.Button("Get Statistics")
                    stats_output = gr.Textbox(label="Statistics", lines=5)
            
            add_button.click(add_transcript, inputs=[new_text, new_metadata], outputs=add_output)
            stats_button.click(get_stats, outputs=stats_output)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            ## AI Agent Chatbot - Technical Details
            
            ### Architecture
            - **LLM**: {Config.OLLAMA_MODEL}
            - **Vector DB**: ChromaDB (Local)
            - **Embeddings**: {Config.EMBEDDING_MODEL}
            - **Voice**: Whisper ({Config.WHISPER_MODEL_SIZE})
            
            ### Features
            1. ✅ Intent Classification
            2. ✅ RAG (Retrieval Augmented Generation)
            3. ✅ Real-time Voice Transcription
            4. ✅ Tool Calling (Search, Sentiment, Statistics)
            5. ✅ Conversation History
            
            ### How It Works
            1. **Input**: Text or voice input from user
            2. **Intent**: Classify user's intention
            3. **RAG**: Search vector database for relevant context
            4. **Tool**: Decide if tools are needed
            5. **Generate**: Create response with LLM
            6. **Output**: Display formatted response
            
            ### Current Database
            - Total Transcripts: {agent.vector_db.get_count()}
            - Embedding Dimensions: 384
            - Search Method: Cosine Similarity
            """)
    
    return interface


def main():
    """Main function to run the chatbot"""
    
    print("=" * 70)
    print("AI Agent Chatbot - Local Open-Source POC".center(70))
    print("=" * 70)
    
    # Initialize agent
    print("\n[1/4] Initializing AI Agent...")
    agent = AIAgent()
    
    # Load sample data
    print("\n[2/4] Loading conversation data...")
    agent.load_sample_data()
    
    # Verify Ollama connection
    print("\n[3/4] Verifying Ollama connection...")
    try:
        ollama.list()
        print("✓ Ollama is running")
    except Exception as e:
        print(f"✗ Ollama not available: {e}")
        print("Please start Ollama: ollama serve")
        return
    
    # Create and launch interface
    print("\n[4/4] Launching Gradio interface...")
    interface = create_gradio_interface(agent)
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!".center(70))
    print("=" * 70)
    print(f"\n🌐 Access the chatbot at: http://localhost:{Config.APP_PORT}")
    print("\n📋 Features Available:")
    print("  • Text-based chat with RAG")
    print("  • Voice input with real-time transcription")
    print("  • Intent classification")
    print("  • Tool calling capabilities")
    print("  • Database management")
    print("\n💡 Tip: Use Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    interface.launch(
        share=False, 
        server_port=Config.APP_PORT,
        show_error=True
    )


if __name__ == "__main__":
    main()

