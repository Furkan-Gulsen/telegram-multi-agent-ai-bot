from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Message:
    user_id: str
    message: str
    timestamp: datetime
    is_processed: bool = False
    is_file: bool = False
    batch_id: Optional[str] = None
    document_ref: Optional[Dict[str, Any]] = None
    document_info: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message instance from a dictionary"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Message instance to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @property
    def is_document_related(self) -> bool:
        """Check if message is document-related"""
        return self.is_file or bool(self.document_ref)
