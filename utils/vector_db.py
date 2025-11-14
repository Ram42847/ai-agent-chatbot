# ============================================================================
# FILE: utils/vector_db.py
# ============================================================================
"""Vector database management using ChromaDB"""

import time
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import Config


class VectorDatabase:
    """Manages ChromaDB for storing and retrieving conversation transcripts"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
        # Initialize embedding model
        print(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"description": "Past conversation transcripts"}
        )
        print(f"Vector database initialized: {self.collection.count()} documents")
        
    def add_transcript(self, text: str, metadata: Dict) -> str:
        """Add a single conversation transcript"""
        embedding = self.embedding_model.encode(text).tolist()
        doc_id = f"doc_{int(time.time() * 1000)}"
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
        return doc_id
        
    def bulk_add_transcripts(self, transcripts: List[Dict]) -> List[str]:
        """Bulk add multiple transcripts"""
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, transcript in enumerate(transcripts):
            text = transcript.get('text', '')
            metadata = transcript.get('metadata', {})

            # Ensure metadata is a dictionary
            if not isinstance(metadata, dict):
                metadata = {"tags": metadata}  # Wrap list or string in a dict

            # Convert any list values inside metadata dict to strings
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ", ".join(value)
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)  # Convert any other type to string

            embedding = self.embedding_model.encode(text).tolist()
            doc_id = f"doc_{int(time.time() * 1000)}_{i}"

            ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Added {len(transcripts)} transcripts to database")
        return ids

        
    def search(self, query: str, n_results: int = None) -> List[Dict]:
        """Search for similar transcripts"""
        n_results = n_results or Config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        return formatted_results
    
    def get_count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()
    
    def delete(self, doc_id: str):
        """Delete a document by ID"""
        self.collection.delete(ids=[doc_id])
