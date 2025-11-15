# ============================================================================
# OPTION 1: Enhanced main.py with Gemini Integration
# ============================================================================
"""
Modified main.py to support both Ollama and Gemini
"""

import json
import time
from typing import Dict, Any, Optional
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

from integrations.gemini_integration import GeminiAgent  # Import Gemini agent


class HybridAIAgent:
    """
    Enhanced AI Agent supporting both Ollama and Gemini
    """
    
    def __init__(self, model_provider: str = "ollama", model_name: str = None):
        """
        Initialize agent with choice of model provider
        
        Args:
            model_provider: "ollama" or "gemini"
            model_name: Specific model name (optional)
        """
        print(f"Initializing AI Agent with {model_provider}...")
        
        # Ensure directories exist
        Config.ensure_directories()
        
        # Store provider choice
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        
        # Initialize shared components
        self.vector_db = VectorDatabase()
        self.voice_transcriber = VoiceTranscriber()
        self.tool_executor = ToolExecutor(self.vector_db)
        self.conversation_history = []
        
        # Initialize provider-specific components
        if self.model_provider == "gemini":
            try:
                self.gemini_agent = GeminiAgent()
                # Use Gemini's native intent classification
                self.intent_classifier = None
                print("✓ Gemini agent initialized")
            except Exception as e:
                print(f"⚠ Gemini initialization failed: {e}")
                print("Falling back to Ollama...")
                self.model_provider = "ollama"
                self._init_ollama()
        else:
            self._init_ollama()
    
    def _init_ollama(self):
        """Initialize Ollama components"""
        self.model_name = self.model_name or Config.OLLAMA_MODEL
        self.intent_classifier = IntentClassifier(self.model_name)
        self.gemini_agent = None
        print("✓ Ollama agent initialized")
    
    def process_query(self, user_input: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Process user query with selected provider
        """
        start_time = time.time()
        
        print(f"\n[Processing with {self.model_provider.upper()}] User query: {user_input[:50]}...")
        
        # Use appropriate processing method
        if self.model_provider == "gemini":
            return self._process_with_gemini(user_input, use_rag, start_time)
        else:
            return self._process_with_ollama(user_input, use_rag, start_time)
    
    def _process_with_gemini(self, user_input: str, use_rag: bool, start_time: float) -> Dict[str, Any]:
        """Process query using Gemini"""
        
        try:
            # Step 1: Get intent using Gemini
            intent_result = self.gemini_agent.classify_intent(user_input)
            intent = intent_result.get('intent', 'general_conversation')
            confidence = intent_result.get('confidence', 0.5)
            print(f"[Intent] {intent} (confidence: {confidence:.2f})")
            
            # Step 2: RAG - Retrieve context if enabled
            context_docs = []
            if use_rag:
                results = self.vector_db.search(user_input, n_results=Config.TOP_K_RESULTS)
                context_docs = [r['document'] for r in results]
                print(f"[RAG] Retrieved {len(results)} relevant documents")
            
            # Step 3: Generate response with Gemini (includes automatic tool calling)
            gemini_result = self.gemini_agent.chat_with_rag(
                user_query=user_input,
                context=context_docs,
                intent=intent
            )
            print("GEMINIT RESULT , ", gemini_result)
            response_text = gemini_result['response']
            tool_calls = gemini_result.get('function_calls', [])
            
            processing_time = time.time() - start_time
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_input,
                "assistant": response_text,
                "intent": intent,
                "timestamp": datetime.now().isoformat(),
                "provider": "gemini"
            })
            
            print(f"[Complete] Processing time: {processing_time:.2f}s")
            
            return {
                "response": response_text,
                "intent": intent,
                "confidence": confidence,
                "tool_used": len(tool_calls) > 0,
                "tool_response": tool_calls,
                "processing_time": processing_time,
                "provider": "gemini"
            }
            
        except Exception as e:
            print(f"[Error] Gemini processing failed: {e}")
            # Fallback to Ollama
            print("[Fallback] Switching to Ollama...")
            return self._process_with_ollama(user_input, use_rag, start_time)
    
    def _process_with_ollama(self, user_input: str, use_rag: bool, start_time: float) -> Dict[str, Any]:
        """Process query using Ollama (original implementation)"""
        
        # Step 1: Intent Classification
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
        
        # Step 3: Build prompt
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
        
        # Step 4: Generate response
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
            if '"use_tool": true' in response_text:
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
                            
                            # Generate final response
                            final_response = ollama.generate(
                                model=self.model_name,
                                prompt=f"User: {user_input}\n\nTool result: {tool_response}\n\nProvide response:",
                                options={"temperature": Config.OLLAMA_TEMPERATURE}
                            )
                            response_text = final_response['response']
                except Exception as e:
                    print(f"[Error] Tool calling error: {e}")
            
            processing_time = time.time() - start_time
            
            # Update history
            self.conversation_history.append({
                "user": user_input,
                "assistant": response_text,
                "intent": intent,
                "timestamp": datetime.now().isoformat(),
                "provider": "ollama"
            })
            
            print(f"[Complete] Processing time: {processing_time:.2f}s")
            
            return {
                "response": response_text,
                "intent": intent,
                "confidence": confidence,
                "tool_used": tool_response is not None,
                "tool_response": tool_response,
                "processing_time": processing_time,
                "provider": "ollama"
            }
            
        except Exception as e:
            print(f"[Error] {str(e)}")
            return {
                "response": f"Error processing your request. Please try again.",
                "intent": intent,
                "confidence": confidence,
                "tool_used": False,
                "tool_response": None,
                "processing_time": time.time() - start_time,
                "provider": "ollama"
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
    
    def switch_provider(self, new_provider: str):
        """Switch between Ollama and Gemini"""
        if new_provider.lower() == self.model_provider:
            return f"Already using {new_provider}"
        
        self.model_provider = new_provider.lower()
        
        if self.model_provider == "gemini":
            try:
                if not self.gemini_agent:
                    self.gemini_agent = GeminiAgent()
                return "✓ Switched to Gemini"
            except Exception as e:
                self.model_provider = "ollama"
                return f"✗ Failed to switch to Gemini: {e}"
        else:
            if not self.intent_classifier:
                self._init_ollama()
            return "✓ Switched to Ollama"
    
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
                "text": "Customer billing issue resolved. Refund processed.",
                "metadata": {"date": "2024-01-15", "category": "billing"}
            },
            {
                "text": "Technical support for login problems. Password reset successful.",
                "metadata": {"date": "2024-01-16", "category": "technical"}
            }
        ]
        
        self.vector_db.bulk_add_transcripts(sample_transcripts)
        with open(f"{Config.DATA_DIR}/sample_transcripts.json", 'w') as f:
            json.dump(sample_transcripts, f, indent=2)


# ============================================================================
# Enhanced Gradio Interface with Provider Selection
# ============================================================================

def create_gradio_interface(agent: HybridAIAgent):
    """Create enhanced Gradio interface with provider selection"""
    
    def chat_response(message, history):
        """Handle chat messages"""
        result = agent.process_query(message)
        response = result['response']
        
        # Add metadata with provider info
        metadata = f"\n\n---\n*Provider: {result['provider'].upper()} | "
        metadata += f"Intent: {result['intent']} | "
        metadata += f"Confidence: {result['confidence']:.2f} | "
        metadata += f"Time: {result['processing_time']:.2f}s*"
        
        if result['tool_used']:
            metadata += f"\n*🔧 Tools used to retrieve information*"
        
        return response + metadata
    
    def voice_response(audio):
        """Handle voice input"""
        if audio is None:
            return "Please record audio first."
        
        result = agent.process_voice_query(audio)
        return result['response']
    
    def switch_model_provider(provider):
        """Switch between Ollama and Gemini"""
        return agent.switch_provider(provider)
    
    def add_transcript(text, metadata_str):
        """Add new transcript"""
        try:
            metadata = json.loads(metadata_str)
            doc_id = agent.vector_db.add_transcript(text, metadata)
            return f"✓ Transcript added! (ID: {doc_id})"
        except Exception as e:
            return f"✗ Error: {str(e)}"
    
    def get_stats():
        """Get database statistics"""
        count = agent.vector_db.get_count()
        provider = agent.model_provider.upper()
        return f"Provider: {provider}\nTotal transcripts: {count}"
    
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
        .provider-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 5px;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="header">
            <h1>🤖 AI Agent Chatbot with RAG</h1>
            <p>Powered by Ollama, Gemini, ChromaDB, and Whisper</p>
            <div>
                <span class="provider-badge" style="background: #4CAF50;">Ollama: Local & Private</span>
                <span class="provider-badge" style="background: #2196F3;">Gemini: Cloud & Powerful</span>
            </div>
        </div>
        """)
        
        # Provider Selection
        with gr.Row():
            provider_selector = gr.Radio(
                choices=["ollama", "gemini"],
                value=agent.model_provider,
                label="🔄 Select LLM Provider",
                info="Switch between local Ollama and cloud Gemini"
            )
            provider_status = gr.Textbox(
                label="Status",
                value=f"Currently using: {agent.model_provider.upper()}",
                interactive=False
            )
        
        provider_selector.change(
            fn=switch_model_provider,
            inputs=provider_selector,
            outputs=provider_status
        )
        
        with gr.Tab("💬 Text Chat"):
            gr.Markdown("""
            ### Chat with the AI Agent
            - Ask questions and get contextual responses
            - Agent automatically searches past conversations
            - Tools are used when needed (search, sentiment, stats)
            """)
            
            chatbot = gr.Chatbot(height=500, show_label=False)
            msg = gr.Textbox(
                label="Your Message", 
                placeholder="Ask anything...",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send 📤", variant="primary")
                clear = gr.Button("Clear 🗑️")
            
            gr.Examples(
                examples=[
                    "What billing issues have we had recently?",
                    "Search for technical support conversations",
                    "Analyze the sentiment of customer feedback",
                    "How many conversations are in the database?"
                ],
                inputs=msg
            )
            
            msg.submit(chat_response, [msg, chatbot], [chatbot])
            submit.click(chat_response, [msg, chatbot], [chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        
        with gr.Tab("🎤 Voice Chat"):
            gr.Markdown("### Voice Input")
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="Record Your Voice"
            )
            voice_output = gr.Textbox(label="Response", lines=10)
            voice_button = gr.Button("Process Voice 🎙️", variant="primary")
            
            voice_button.click(voice_response, inputs=audio_input, outputs=voice_output)
        
        with gr.Tab("📊 Database Management"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Add New Transcript")
                    new_text = gr.Textbox(
                        label="Transcript Text", 
                        lines=5,
                        placeholder="Enter conversation transcript..."
                    )
                    new_metadata = gr.Textbox(
                        label="Metadata (JSON)", 
                        lines=3,
                        value='{"date": "2024-01-20", "agent": "Agent Name", "category": "general"}'
                    )
                    add_button = gr.Button("Add Transcript ➕", variant="primary")
                    add_output = gr.Textbox(label="Status", lines=2)
                
                with gr.Column():
                    gr.Markdown("### Database Statistics")
                    stats_button = gr.Button("Get Statistics 📈")
                    stats_output = gr.Textbox(label="Statistics", lines=8)
            
            add_button.click(add_transcript, inputs=[new_text, new_metadata], outputs=add_output)
            stats_button.click(get_stats, outputs=stats_output)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            ## AI Agent Chatbot - Hybrid Architecture
            
            ### Available Providers
            
            #### 🟢 Ollama (Local)
            - **Model**: {Config.OLLAMA_MODEL}
            - **Pros**: 100% private, no API costs, works offline
            - **Cons**: Requires local resources, slower on CPU
            
            #### 🔵 Gemini (Cloud)
            - **Model**: {Config.GEMINI_MODEL}
            - **Pros**: Faster, more powerful, native tool calling
            - **Cons**: Requires API key, costs per request
            
            ### Architecture Components
            - **Vector DB**: ChromaDB (Local, Persistent)
            - **Embeddings**: {Config.EMBEDDING_MODEL} (384 dimensions)
            - **Voice**: Whisper ({Config.WHISPER_MODEL_SIZE})
            - **UI**: Gradio
            
            ### Current Status
            - **Active Provider**: {agent.model_provider.upper()}
            - **Database Size**: {agent.vector_db.get_count()} transcripts
            - **Embedding Dimensions**: 384
            
            ### How to Switch Providers
            1. Select provider from dropdown at top
            2. For Gemini: Ensure GOOGLE_API_KEY is set in .env
            3. System automatically switches and maintains conversation history
            """)
    
    return interface


# ============================================================================
# Main Function with Provider Selection
# ============================================================================

def main():
    """Main function with provider selection"""
    
    import sys
    
    print("=" * 70)
    print("AI Agent Chatbot - Hybrid Provider Support".center(70))
    print("=" * 70)
    
    # Check for provider argument
    provider = "ollama"
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider not in ["ollama", "gemini"]:
            print(f"Invalid provider: {provider}")
            print("Usage: python main.py [ollama|gemini]")
            return
    
    print(f"\n[1/4] Initializing AI Agent with {provider.upper()}...")
    agent = HybridAIAgent(model_provider=provider)
    
    print("\n[2/4] Loading conversation data...")
    agent.load_sample_data()
    
    print("\n[3/4] Verifying providers...")
    
    # Check Ollama
    try:
        ollama.list()
        print("✓ Ollama is available")
    except:
        print("⚠ Ollama not available")
    
    # Check Gemini
    if Config.GOOGLE_API_KEY:
        print("✓ Gemini API key configured")
    else:
        print("⚠ Gemini API key not found (set GOOGLE_API_KEY in .env)")
    
    print("\n[4/4] Launching Gradio interface...")
    interface = create_gradio_interface(agent)
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!".center(70))
    print("=" * 70)
    print(f"\n🌐 Access: http://localhost:{Config.APP_PORT}")
    print(f"🤖 Provider: {provider.upper()}")
    print("\n💡 Switch providers anytime using the dropdown in the UI")
    print("=" * 70 + "\n")
    
    interface.launch(
        share=False, 
        server_port=Config.APP_PORT,
        show_error=True
    )


if __name__ == "__main__":
    main()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
# Run with Ollama (default)
python main.py

# Run with Gemini
python main.py gemini

# Programmatic usage
from main import HybridAIAgent

# Use Ollama
agent = HybridAIAgent(model_provider="ollama")
result = agent.process_query("What are common billing issues?")

# Use Gemini
agent = HybridAIAgent(model_provider="gemini")
result = agent.process_query("Analyze customer sentiment")

# Switch providers at runtime
agent.switch_provider("gemini")
"""