from typing import Dict, Any, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.operations import IndexModel
import numpy as np
from ..config import (
    MONGODB_URI, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS,
    VECTOR_DIMENSIONS,
    VECTOR_SIMILARITY,
    VECTOR_INDEX_NAME
)

class MongoDB:
    def __init__(self):
        self.client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,  # Disable certificate verification for development
            serverSelectionTimeoutMS=5000  # 5 second timeout
        )
        self.db = self.client[MONGODB_DB_NAME]
        self.message_queue = self.db[MONGODB_COLLECTIONS["messages"]]
        self.documents = self.db[MONGODB_COLLECTIONS["documents"]]
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Create indexes for better query performance"""
        # Message queue indexes
        self.message_queue.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
        self.message_queue.create_index([("is_processed", ASCENDING)])
        
        # Document indexes
        self.documents.create_index([("user_id", ASCENDING)])
        self.documents.create_index([("file_hash", ASCENDING)])
        self.documents.create_index([("chunks.metadata.user_id", ASCENDING)])
        
        # Vector search index for embeddings
        try:
            # Drop existing vector index if exists
            try:
                self.documents.drop_index(VECTOR_INDEX_NAME)
            except:
                pass
            
            # Create new vector search index
            self.documents.create_index(
                [("chunks.embedding", "vector")],
                {
                    "name": VECTOR_INDEX_NAME,
                    "vectorSearchOptions": {
                        "numDimensions": VECTOR_DIMENSIONS,
                        "similarity": VECTOR_SIMILARITY
                    }
                }
            )
            print("Vector search index created successfully")
        except Exception as e:
            print(f"Error creating vector search index: {str(e)}")
            print("Attempting to create basic indexes for fallback...")
            try:
                # Create basic indexes for fallback
                self.documents.create_index([("chunks.content", "text")])
                print("Basic indexes created successfully")
            except Exception as inner_e:
                print(f"Error creating basic indexes: {str(inner_e)}")
    
    def add_message(self, user_id: str, message: str, is_file: bool = False, **kwargs) -> str:
        """Add a message to the queue"""
        message_data = {
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.utcnow(),
            "is_processed": False,
            "is_file": is_file,
            "batch_id": None,
            **kwargs
        }
        result = self.message_queue.insert_one(message_data)
        return str(result.inserted_id)
    
    def get_pending_messages(self, user_id: str, cutoff_time: datetime, limit: int) -> List[Dict[str, Any]]:
        """Get pending messages for processing"""
        return list(self.message_queue.find({
            "user_id": user_id,
            "is_processed": False,
            "timestamp": {"$gte": cutoff_time}
        }).sort("timestamp", ASCENDING).limit(limit))
    
    def mark_messages_as_processed(self, message_ids: List[str], batch_id: str):
        """Mark messages as processed"""
        self.message_queue.update_many(
            {"_id": {"$in": message_ids}},
            {
                "$set": {
                    "is_processed": True,
                    "batch_id": batch_id,
                    "processing_started": datetime.utcnow()
                }
            }
        )
    
    def update_message_response(self, batch_id: str, response: str):
        """Update messages with response"""
        self.message_queue.update_many(
            {"batch_id": batch_id},
            {
                "$set": {
                    "processing_completed": datetime.utcnow(),
                    "response": response
                }
            }
        )
    
    def add_document(self, doc_info: Dict[str, Any]) -> str:
        """Add a document to the database"""
        # Convert numpy arrays to lists for MongoDB storage
        if "chunks" in doc_info:
            for chunk in doc_info["chunks"]:
                if "embedding" in chunk and isinstance(chunk["embedding"], np.ndarray):
                    chunk["embedding"] = chunk["embedding"].tolist()
        
        result = self.documents.insert_one(doc_info)
        return str(result.inserted_id)
    
    def get_document_by_hash(self, file_hash: str) -> Dict[str, Any]:
        """Get document by file hash"""
        return self.documents.find_one({"file_hash": file_hash})
    
    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        return list(self.documents.find({"user_id": user_id}))
    
    def search_similar_chunks(self, query_vector: List[float], user_id: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Convert query vector to list if it's numpy array
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Use MongoDB Atlas vector search
            pipeline = [
                {
                    "$search": {
                        "index": VECTOR_INDEX_NAME,
                        "knnBeta": {
                            "vector": query_vector,
                            "path": "chunks.embedding",
                            "k": k * 2,  # Get more results initially for better filtering
                            "filter": {
                                "equals": {
                                    "path": "chunks.metadata.user_id",
                                    "value": user_id
                                }
                            }
                        }
                    }
                },
                {"$unwind": "$chunks"},
                {
                    "$match": {
                        "chunks.metadata.user_id": user_id
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "content": "$chunks.content",
                        "metadata": "$chunks.metadata",
                        "score": {"$meta": "searchScore"}
                    }
                },
                {"$sort": {"score": -1}},
                {"$limit": k}
            ]

            results = list(self.documents.aggregate(pipeline))
            
            if not results:
                # Fallback to basic similarity search without vector operations
                basic_pipeline = [
                    {"$unwind": "$chunks"},
                    {
                        "$match": {
                            "chunks.metadata.user_id": user_id
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "content": "$chunks.content",
                            "metadata": "$chunks.metadata",
                            "score": 1
                        }
                    },
                    {"$limit": k}
                ]
                
                results = list(self.documents.aggregate(basic_pipeline))
            
            # Process results to ensure uniqueness and quality
            unique_results = []
            seen_content = set()
            
            for result in results:
                content = result.get("content", "").strip()
                if content and content not in seen_content:
                    seen_content.add(content)
                    unique_results.append(result)
            
            return unique_results[:k]
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            # If vector search fails completely, try simple document retrieval
            try:
                basic_results = list(self.documents.aggregate([
                    {"$unwind": "$chunks"},
                    {"$match": {"chunks.metadata.user_id": user_id}},
                    {"$limit": k},
                    {
                        "$project": {
                            "_id": 0,
                            "content": "$chunks.content",
                            "metadata": "$chunks.metadata",
                            "score": 1
                        }
                    }
                ]))
                return basic_results
            except Exception as inner_e:
                print(f"Error in fallback search: {str(inner_e)}")
                return []

# Create a singleton instance
db = MongoDB()
