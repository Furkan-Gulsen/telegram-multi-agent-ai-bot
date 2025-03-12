from typing import Dict, Any, List
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document as LangchainDocument
import numpy as np
import os
import mimetypes
from langchain.prompts import ChatPromptTemplate

from ..config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_INDEX_NAME, VECTOR_DIMENSIONS
from ..database.mongodb import db
from ..models.document import Document

class DocumentHandler:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        self.db = db  # Initialize MongoDB connection
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=VECTOR_DIMENSIONS
        )
        mimetypes.init()
    
    def _get_loader(self, file_path: str):
        """Get appropriate loader based on file type"""
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type == 'application/pdf':
            return PyPDFLoader(file_path)
        elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return Docx2txtLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)
    
    async def process_document(self, file_path: str, user_id: str) -> Dict[str, Any]:
        """Process a document and store it with embeddings in MongoDB"""
        try:
            # Create document instance
            document = Document.create(user_id, file_path)
            
            # Check if document already exists
            existing_doc = self.db.get_document_by_hash(document.file_hash)
            if existing_doc:
                return {"status": "exists", "doc_id": existing_doc["_id"]}
            
            # Load and split document
            loader = self._get_loader(file_path)
            raw_documents = loader.load()
            
            # Combine all text from the document
            full_text = "\n\n".join([doc.page_content for doc in raw_documents])
            
            # Split into chunks with better context preservation
            chunks = self.text_splitter.create_documents(
                texts=[full_text],
                metadatas=[{
                    "user_id": user_id,
                    "file_hash": document.file_hash,
                    "file_name": os.path.basename(file_path),
                    "source": file_path
                }]
            )
            
            # Update document with chunk information
            document.chunk_count = len(chunks)
            
            # Get embeddings for all chunks in batches
            chunk_texts = [chunk.page_content for chunk in chunks]
            all_embeddings = []
            
            # Process in batches of 100
            batch_size = 100
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            
            # Create chunks with embeddings and better metadata
            chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                chunk_id = f"{document.file_hash}_{i}"
                chunk_text = chunk.page_content.strip()
                
                # Skip empty chunks
                if not chunk_text:
                    continue
                
                # Create chunk metadata
                chunk_metadata = {
                    "user_id": user_id,
                    "file_hash": document.file_hash,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "total_chunks": len(chunks),
                    "char_length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "created_at": datetime.utcnow()
                }
                
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "embedding": embedding,
                    "metadata": chunk_metadata
                })
            
            # Update document with chunks data
            document.chunks = chunks_data
            document.status = "processed"
            
            # Store document in MongoDB
            doc_id = self.db.add_document(document.to_dict())
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "file_hash": document.file_hash,
                "chunk_count": len(chunks_data),
                "metadata": document.metadata
            }
            
        except Exception as e:
            error_info = {
                "user_id": user_id,
                "file_path": file_path,
                "error": str(e),
                "timestamp": datetime.utcnow(),
                "status": "failed"
            }
            self.db.add_document(error_info)
            raise
    
    async def query_documents(self, query: str, user_id: str, k: int = 5) -> Dict[str, Any]:
        """Query documents using MongoDB vector search"""
        try:
            # Get user's documents from MongoDB
            user_docs = self.db.get_user_documents(user_id)
            if not user_docs:
                return {
                    "answer": "No documents found in your collection.",
                    "sources": []
                }

            # Generate semantic variations for better search coverage
            semantic_variations = [
                query,  # Original query
                f"find information about {query}",  # Explicit search
                f"what does the document say about {query}",  # Document-focused
                f"find content related to {query}",  # Related content
                query.replace("?", "").strip(),  # Clean query
                f"extract information about {query}"  # Information extraction
            ]
            
            all_chunks = []
            seen_chunk_ids = set()
            
            # Search with each variation and collect results
            for variation in semantic_variations:
                # Get embedding for the variation
                query_embedding = self.embeddings.embed_query(variation)
                
                # Search similar chunks
                chunks = self.db.search_similar_chunks(query_embedding, user_id, k=5)
                
                # Add unique chunks to results
                for chunk in chunks:
                    chunk_id = f"{chunk['metadata'].get('file_hash', '')}_{chunk['metadata'].get('chunk_index', '')}"
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_chunks.append(chunk)
            
            if not all_chunks:
                return {
                    "answer": "No relevant content found in your documents.",
                    "sources": []
                }
            
            # Sort chunks by similarity score
            all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Take top k most relevant chunks
            top_chunks = all_chunks[:k]
            
            # Create context with full content
            contexts = []
            sources = []
            
            for chunk in top_chunks:
                metadata = chunk["metadata"]
                score = chunk.get("score", 0)
                
                # Create Langchain Document objects
                doc = LangchainDocument(
                    page_content=chunk["content"],
                    metadata={
                        "source": metadata.get("file_name", "Unknown"),
                        "section": metadata.get("chunk_index", 0) + 1,
                        "score": score
                    }
                )
                contexts.append(doc)
                
                # Add to sources with full content
                sources.append({
                    "content": chunk["content"],
                    "metadata": metadata,
                    "similarity_score": score
                })
            
            # Create context string from documents
            context_str = "\n\n".join([
                f"From {doc.metadata['source']} (Section {doc.metadata['section']}):\n{doc.page_content}"
                for doc in contexts
            ])

            # Create prompt template
            from langchain.prompts import PromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            
            prompt = PromptTemplate(
                template="""You are a knowledgeable assistant providing clear and concise information.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Answer directly and naturally, as if you inherently know the information
            2. Don't say phrases like "According to the document" or "I found in the documents"
            3. Don't mention sources unless specifically asked
            4. Keep the answer focused and to the point
            5. Use a conversational but professional tone
            6. If information isn't available, say so briefly and clearly
            7. Avoid repeating information
            8. If responding in Turkish, use ASCII characters (replace: ğ->g, ü->u, ş->s, ı->i, ö->o, ç->c)

            Answer:""",
                input_variables=["context", "question"]
            )
            
            # Create chain using the new API
            llm = ChatOpenAI(temperature=0.3)
            chain = (
                {
                    "context": lambda x: context_str,
                    "question": lambda x: x["question"]
                }
                | prompt
                | llm
            )
            
            # Get response using the new invoke method
            response = chain.invoke({"question": query})
            answer = response.content if hasattr(response, 'content') else str(response)
            

            return {
                "answer": answer,
                "sources": sources,
                "total_docs": len(user_docs),
                "docs_used": len(seen_chunk_ids)
            }

        except Exception as e:
            print(f"Error in query_documents: {str(e)}")
            # Fallback to direct content return
            if all_chunks:
                answer = "Here's what I found in the documents:\n\n"
                for chunk in all_chunks[:3]:
                    answer += f"- {chunk['content']}\n\n"
                return {
                    "answer": answer,
                    "sources": all_chunks[:3],
                    "total_docs": len(user_docs),
                    "docs_used": len(all_chunks)
                }
            return {
                "answer": f"Error querying documents: {str(e)}",
                "sources": []
            }

# Create a singleton instance
document_handler = DocumentHandler()
