from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import os
import mimetypes

@dataclass
class Document:
    user_id: str
    file_path: str
    file_hash: str
    upload_time: datetime
    chunk_count: int
    status: str
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a Document instance from a dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Document instance to dictionary"""
        return self.__dict__
    
    @staticmethod
    def calculate_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def get_metadata(file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        return {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "mime_type": mimetypes.guess_type(file_path)[0]
        }
    
    @classmethod
    def create(cls, user_id: str, file_path: str) -> 'Document':
        """Create a new Document instance"""
        file_hash = cls.calculate_hash(file_path)
        metadata = cls.get_metadata(file_path)
        
        return cls(
            user_id=user_id,
            file_path=file_path,
            file_hash=file_hash,
            upload_time=datetime.utcnow(),
            chunk_count=0,  # Will be updated after processing
            status="uploaded",
            metadata=metadata,
            chunks=[]  # Will be populated with chunks and their embeddings
        )
