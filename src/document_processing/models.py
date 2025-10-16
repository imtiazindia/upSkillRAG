"""
Data models for document processing components
Defines contracts for PDF processing, text chunking, and metadata extraction
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path

class ProcessingStatus(Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    FAILED = "failed"

class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    UNKNOWN = "unknown"

@dataclass
class DocumentMetadata:
    """Metadata extracted from a document"""
    file_path: Path
    file_name: str
    file_size: int  # in bytes
    document_type: DocumentType
    page_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: Optional[str] = None
    extracted_metadata: Optional[Dict[str, Any]] = None
    
    @property
    def file_extension(self) -> str:
        """Get file extension in lowercase"""
        return self.file_path.suffix.lower()

@dataclass
class PageText:
    """Text extracted from a single page with positional information"""
    page_number: int
    text: str
    bbox: Optional[Dict[str, float]] = None  # bounding box coordinates
    sections: Optional[List[str]] = None  # section headers on this page
    images_count: int = 0
    tables_count: int = 0
    
    @property
    def word_count(self) -> int:
        """Calculate word count for the page"""
        return len(self.text.split())

@dataclass
class ExtractedDocument:
    """Complete document after text extraction"""
    metadata: DocumentMetadata
    pages: List[PageText]
    processing_time: float
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_message: Optional[str] = None
    
    @property
    def total_word_count(self) -> int:
        """Total words across all pages"""
        return sum(page.word_count for page in self.pages)
    
    @property
    def full_text(self) -> str:
        """Combine text from all pages"""
        return "\n\n".join(page.text for page in self.pages)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 1000  # target chunk size in characters
    chunk_overlap: int = 100  # overlap between chunks in characters
    max_chunk_size: int = 1500  # maximum chunk size
    min_chunk_size: int = 200  # minimum chunk size
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    section_aware: bool = True
    
    def validate(self) -> bool:
        """Validate chunking configuration"""
        if self.chunk_size <= 0:
            return False
        if self.chunk_overlap >= self.chunk_size:
            return False
        if self.max_chunk_size < self.chunk_size:
            return False
        if self.min_chunk_size > self.chunk_size:
            return False
        return True

@dataclass
class TextChunk:
    """A chunk of text from a document with metadata"""
    content: str
    chunk_id: str
    document_id: str
    page_numbers: List[int]
    section_titles: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: int = 0  # character count
    word_count: int = 0
    position_in_document: int = 0  # order in original document
    
    def __post_init__(self):
        """Calculate sizes after initialization"""
        self.chunk_size = len(self.content)
        self.word_count = len(self.content.split())
    
    @property
    def citation_info(self) -> str:
        """Generate citation information for this chunk"""
        pages = f"Pages {min(self.page_numbers)}-{max(self.page_numbers)}" if self.page_numbers else "Unknown pages"
        sections = f" - {', '.join(self.section_titles)}" if self.section_titles else ""
        return f"{self.document_id} - {pages}{sections}"

@dataclass
class ChunkedDocument:
    """Document after chunking process"""
    extracted_document: ExtractedDocument
    chunks: List[TextChunk]
    chunking_config: ChunkingConfig
    processing_time: float
    chunks_generated: int = 0
    average_chunk_size: float = 0.0
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Calculate chunk statistics after initialization"""
        self.chunks_generated = len(self.chunks)
        if self.chunks:
            self.average_chunk_size = sum(chunk.chunk_size for chunk in self.chunks) / len(self.chunks)

@dataclass
class ProcessingProgress:
    """Progress tracking for document processing"""
    document_id: str
    file_name: str
    current_stage: str
    progress_percentage: float  # 0.0 to 100.0
    estimated_time_remaining: Optional[float] = None  # in seconds
    current_operation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ProcessingResult:
    """Final result of document processing"""
    document_id: str
    file_path: Path
    status: ProcessingStatus
    extracted_document: Optional[ExtractedDocument] = None
    chunked_document: Optional[ChunkedDocument] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return self.status == ProcessingStatus.COMPLETED and self.chunked_document is not None
    
    @property
    def total_chunks(self) -> int:
        """Get total number of chunks generated"""
        return len(self.chunked_document.chunks) if self.chunked_document else 0

@dataclass
class BatchProcessingConfig:
    """Configuration for batch document processing"""
    max_concurrent_processes: int = 2
    chunking_config: ChunkingConfig = None
    enable_metadata_extraction: bool = True
    enable_progress_tracking: bool = True
    output_directory: Optional[Path] = None
    fail_fast: bool = False  # Stop on first error
    
    def __post_init__(self):
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig()

@dataclass
class BatchProcessingResult:
    """Results of batch document processing"""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_processing_time: float
    results: List[ProcessingResult]
    start_time: datetime
    end_time: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_documents == 0:
            return 0.0
        return (self.successful_documents / self.total_documents) * 100.0
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per document"""
        if self.total_documents == 0:
            return 0.0
        return self.total_processing_time / self.total_documents
    
    @property
    def total_chunks_generated(self) -> int:
        """Total chunks generated across all documents"""
        return sum(result.total_chunks for result in self.results if result.is_successful)
