"""
Text Chunker for intelligent text splitting
Implements multiple chunking strategies with semantic awareness
"""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from .models import (
    ExtractedDocument, TextChunk, ChunkedDocument, ChunkingConfig,
    ChunkingStrategy, ProcessingStatus, ProcessingProgress
)

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Implements intelligent text chunking with multiple strategies
    Handles semantic boundaries, overlap, and document structure preservation
    """
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize Text Chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        self._validate_config()
        
        # Boundary patterns
        self.sentence_endings = r'[.!?]+'
        self.paragraph_separators = r'\n\s*\n'
        self.section_indicators = [
            r'^\s*(?:chapter|section|part)\s+[IVXLCDM0-9]',
            r'^\s*\d+\.\s+[A-Z]',
            r'^\s*[A-Z][A-Z\s]{10,}',
        ]
        
        logger.info(f"Text Chunker initialized with strategy: {config.strategy.value}")
    
    def _validate_config(self) -> None:
        """Validate chunking configuration"""
        if not self.config.validate():
            raise ValueError("Invalid chunking configuration")
    
    def chunk_document(self, 
                      extracted_doc: ExtractedDocument,
                      progress_callback: Optional[callable] = None) -> ChunkedDocument:
        """
        Chunk an extracted document based on configuration
        
        Args:
            extracted_doc: Extracted document to chunk
            progress_callback: Optional progress callback
            
        Returns:
            ChunkedDocument: Chunked document result
        """
        start_time = time.time()
        logger.info(f"Starting document chunking: {extracted_doc.metadata.file_name}")
        
        try:
            # Update progress
            self._update_progress(progress_callback, "preparing", 10, extracted_doc.metadata.file_name)
            
            # Choose chunking strategy
            if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
                chunks = self._fixed_size_chunking(extracted_doc, progress_callback)
            elif self.config.strategy == ChunkingStrategy.SEMANTIC:
                chunks = self._semantic_chunking(extracted_doc, progress_callback)
            else:  # HYBRID
                chunks = self._hybrid_chunking(extracted_doc, progress_callback)
            
            # Update progress
            self._update_progress(progress_callback, "finalizing", 90, extracted_doc.metadata.file_name)
            
            # Apply overlap if configured
            if self.config.chunk_overlap > 0:
                chunks = self._apply_overlap(chunks)
            
            # Filter chunks by size constraints
            chunks = self._filter_chunks_by_size(chunks)
            
            # Assign positions and IDs
            chunks = self._finalize_chunks(chunks, extracted_doc.metadata.file_path.stem)
            
            processing_time = time.time() - start_time
            
            result = ChunkedDocument(
                extracted_document=extracted_doc,
                chunks=chunks,
                chunking_config=self.config,
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
            self._update_progress(progress_callback, "completed", 100, extracted_doc.metadata.file_name)
            logger.info(f"Chunking completed: {len(chunks)} chunks generated")
            
            return result
            
        except Exception as e:
            logger.error(f"Chunking failed for {extracted_doc.metadata.file_name}: {e}")
            return ChunkedDocument(
                extracted_document=extracted_doc,
                chunks=[],
                chunking_config=self.config,
                processing_time=time.time() - start_time,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    def _fixed_size_chunking(self, 
                           extracted_doc: ExtractedDocument,
                           progress_callback: Optional[callable] = None) -> List[TextChunk]:
        """
        Fixed-size chunking strategy
        
        Args:
            extracted_doc: Extracted document
            progress_callback: Progress callback
            
        Returns:
            List of text chunks
        """
        full_text = extracted_doc.full_text
        chunks = []
        position = 0
        
        while position < len(full_text):
            # Calculate chunk end position
            chunk_end = position + self.config.chunk_size
            
            # Adjust chunk end to respect boundaries if configured
            if self.config.respect_sentence_boundaries:
                chunk_end = self._find_sentence_boundary(full_text, chunk_end)
            
            # Extract chunk
            chunk_text = full_text[position:chunk_end].strip()
            
            if chunk_text:
                # Determine page numbers for this chunk
                page_numbers = self._find_pages_for_text(chunk_text, extracted_doc.pages)
                
                # Create chunk
                chunk = TextChunk(
                    content=chunk_text,
                    chunk_id=f"chunk_{len(chunks)}",
                    document_id=extracted_doc.metadata.file_path.stem,
                    page_numbers=page_numbers,
                    position_in_document=len(chunks)
                )
                chunks.append(chunk)
            
            # Move to next position
            position = chunk_end
            
            # Update progress
            if progress_callback and len(chunks) % 10 == 0:
                progress = 20 + (position / len(full_text)) * 60
                self._update_progress(progress_callback, "fixed_chunking", progress, 
                                    f"Chunk {len(chunks)}")
        
        return chunks
    
    def _semantic_chunking(self, 
                         extracted_doc: ExtractedDocument,
                         progress_callback: Optional[callable] = None) -> List[TextChunk]:
        """
        Semantic chunking strategy based on document structure
        
        Args:
            extracted_doc: Extracted document
            progress_callback: Progress callback
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Process by pages first
        for page_num, page in enumerate(extracted_doc.pages, 1):
            if not page.text.strip():
                continue
            
            # Split page text into semantic units
            page_chunks = self._split_into_semantic_units(
                page.text, 
                page_num,
                extracted_doc.metadata.file_path.stem
            )
            
            chunks.extend(page_chunks)
            
            # Update progress
            if progress_callback:
                progress = 20 + (page_num / len(extracted_doc.pages)) * 60
                self._update_progress(progress_callback, "semantic_chunking", progress, 
                                    f"Page {page_num}")
        
        # Merge small chunks and split large ones
        chunks = self._balance_chunk_sizes(chunks)
        
        return chunks
    
    def _hybrid_chunking(self, 
                       extracted_doc: ExtractedDocument,
                       progress_callback: Optional[callable] = None) -> List[TextChunk]:
        """
        Hybrid chunking strategy - semantic first, then fixed-size if needed
        
        Args:
            extracted_doc: Extracted document
            progress_callback: Progress callback
            
        Returns:
            List of text chunks
        """
        # First attempt semantic chunking
        semantic_chunks = self._semantic_chunking(extracted_doc, progress_callback)
        
        # Check if any chunks need further splitting
        final_chunks = []
        
        for i, chunk in enumerate(semantic_chunks):
            if chunk.chunk_size <= self.config.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large semantic chunks using fixed-size approach
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            
            # Update progress
            if progress_callback and i % 10 == 0:
                progress = 20 + (i / len(semantic_chunks)) * 60
                self._update_progress(progress_callback, "hybrid_chunking", progress, 
                                    f"Processing chunk {i}")
        
        return final_chunks
    
    def _split_into_semantic_units(self, 
                                 text: str, 
                                 page_num: int,
                                 document_id: str) -> List[TextChunk]:
        """
        Split text into semantic units (paragraphs, sections)
        
        Args:
            text: Text to split
            page_num: Source page number
            document_id: Document identifier
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        
        # First, split by paragraphs
        paragraphs = re.split(self.paragraph_separators, text)
        
        for para_num, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if paragraph is a section header
            is_section = self._is_section_header(paragraph)
            
            if is_section and chunks:
                # Add section title to previous chunk
                last_chunk = chunks[-1]
                if last_chunk.section_titles is None:
                    last_chunk.section_titles = []
                last_chunk.section_titles.append(paragraph)
            else:
                # Create new chunk for paragraph
                chunk = TextChunk(
                    content=paragraph,
                    chunk_id=f"page_{page_num}_para_{para_num}",
                    document_id=document_id,
                    page_numbers=[page_num],
                    section_titles=[paragraph] if is_section else None,
                    position_in_document=len(chunks)
                )
                
                # Add metadata
                chunk.metadata = {
                    "chunking_strategy": "semantic",
                    "is_section_header": is_section,
                    "paragraph_number": para_num
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def _is_section_header(self, text: str) -> bool:
        """
        Check if text appears to be a section header
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if likely a section header
        """
        if len(text) > 200:  # Too long for a header
            return False
        
        lines = text.split('\n')
        if len(lines) > 3:  # Too many lines for a header
            return False
        
        # Check for section indicator patterns
        for pattern in self.section_indicators:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        
        # Check formatting clues
        if (text.isupper() or  # All caps
            (len(text.split()) <= 10 and text.endswith(':')) or  # Short with colon
            re.search(r'^\d+\.\s', text)):  # Numbered item
            return True
        
        return False
    
    def _find_sentence_boundary(self, text: str, proposed_end: int) -> int:
        """
        Find appropriate sentence boundary near proposed end
        
        Args:
            text: Full text
            proposed_end: Proposed chunk end position
            
        Returns:
            int: Adjusted end position
        """
        if proposed_end >= len(text):
            return len(text)
        
        # Look for sentence endings near proposed end
        search_start = max(0, proposed_end - 100)  # Search 100 chars before
        search_end = min(len(text), proposed_end + 50)  # And 50 chars after
        
        search_text = text[search_start:search_end]
        
        # Find all sentence endings in search area
        matches = list(re.finditer(self.sentence_endings, search_text))
        
        if matches:
            # Use the last sentence ending before proposed end
            for match in reversed(matches):
                boundary_pos = search_start + match.end()
                if boundary_pos <= proposed_end + 10:  # Allow small overshoot
                    return boundary_pos
        
        return proposed_end
    
    def _find_pages_for_text(self, chunk_text: str, pages: List[Any]) -> List[int]:
        """
        Find which pages contain the chunk text
        
        Args:
            chunk_text: Chunk text
            pages: List of page objects
            
        Returns:
            List of page numbers
        """
        page_numbers = []
        sample_text = chunk_text[:100]  # Use first 100 chars for matching
        
        for page in pages:
            if sample_text in page.text:
                page_numbers.append(page.page_number)
        
        return page_numbers if page_numbers else [1]  # Default to page 1 if no match
    
    def _balance_chunk_sizes(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Balance chunk sizes by merging small chunks and splitting large ones
        
        Args:
            chunks: Input chunks
            
        Returns:
            List of balanced chunks
        """
        balanced_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue
            
            # Check if we should merge with current chunk
            combined_size = current_chunk.chunk_size + chunk.chunk_size
            
            if (combined_size <= self.config.chunk_size and
                self._should_merge_chunks(current_chunk, chunk)):
                
                # Merge chunks
                current_chunk = self._merge_chunks(current_chunk, chunk)
            
            else:
                # Add current chunk and start new one
                balanced_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Add the last chunk
        if current_chunk is not None:
            balanced_chunks.append(current_chunk)
        
        # Split any chunks that are still too large
        final_chunks = []
        for chunk in balanced_chunks:
            if chunk.chunk_size > self.config.max_chunk_size:
                split_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _should_merge_chunks(self, chunk1: TextChunk, chunk2: TextChunk) -> bool:
        """
        Determine if two chunks should be merged
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            bool: True if chunks should be merged
        """
        # Don't merge section headers with regular content
        if (chunk1.metadata and chunk1.metadata.get('is_section_header') or
            chunk2.metadata and chunk2.metadata.get('is_section_header')):
            return False
        
        # Check if chunks are from consecutive positions
        if abs(chunk1.position_in_document - chunk2.position_in_document) > 1:
            return False
        
        return True
    
    def _merge_chunks(self, chunk1: TextChunk, chunk2: TextChunk) -> TextChunk:
        """
        Merge two chunks into one
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            TextChunk: Merged chunk
        """
        # Combine content
        merged_content = chunk1.content + "\n\n" + chunk2.content
        
        # Combine page numbers
        merged_pages = list(set(chunk1.page_numbers + chunk2.page_numbers))
        merged_pages.sort()
        
        # Combine section titles
        merged_sections = []
        if chunk1.section_titles:
            merged_sections.extend(chunk1.section_titles)
        if chunk2.section_titles:
            merged_sections.extend(chunk2.section_titles)
        
        # Create merged chunk
        merged_chunk = TextChunk(
            content=merged_content,
            chunk_id=f"merged_{chunk1.chunk_id}_{chunk2.chunk_id}",
            document_id=chunk1.document_id,
            page_numbers=merged_pages,
            section_titles=merged_sections if merged_sections else None,
            position_in_document=chunk1.position_in_document,
            metadata={
                "merged_from": [chunk1.chunk_id, chunk2.chunk_id],
                "chunking_strategy": "merged"
            }
        )
        
        return merged_chunk
    
    def _split_large_chunk(self, chunk: TextChunk) -> List[TextChunk]:
        """
        Split a large chunk into smaller ones
        
        Args:
            chunk: Large chunk to split
            
        Returns:
            List of smaller chunks
        """
        chunks = []
        text = chunk.content
        position = 0
        
        while position < len(text):
            chunk_end = position + self.config.chunk_size
            
            # Respect sentence boundaries
            if self.config.respect_sentence_boundaries:
                chunk_end = self._find_sentence_boundary(text, chunk_end)
            
            chunk_text = text[position:chunk_end].strip()
            
            if chunk_text:
                split_chunk = TextChunk(
                    content=chunk_text,
                    chunk_id=f"{chunk.chunk_id}_part{len(chunks)}",
                    document_id=chunk.document_id,
                    page_numbers=chunk.page_numbers,
                    section_titles=chunk.section_titles,
                    position_in_document=chunk.position_in_document + len(chunks),
                    metadata={
                        "split_from": chunk.chunk_id,
                        "chunking_strategy": "fixed_size_split"
                    }
                )
                chunks.append(split_chunk)
            
            position = chunk_end
        
        return chunks
    
    def _apply_overlap(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Apply overlap between consecutive chunks
        
        Args:
            chunks: Input chunks
            
        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            current_chunk = chunks[i]
            
            if i > 0:
                # Add overlap from previous chunk
                previous_chunk = chunks[i-1]
                overlap_text = self._get_overlap_text(previous_chunk.content, self.config.chunk_overlap)
                if overlap_text:
                    current_chunk.content = overlap_text + "\n\n" + current_chunk.content
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Get overlap text from the end of a chunk
        
        Args:
            text: Source text
            overlap_size: Desired overlap size in characters
            
        Returns:
            str: Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Find a good breaking point (sentence boundary)
        overlap_end = len(text)
        overlap_start = max(0, overlap_end - overlap_size - 50)  # Search area
        
        search_text = text[overlap_start:overlap_end]
        matches = list(re.finditer(self.sentence_endings, search_text))
        
        if matches:
            # Use the last sentence ending
            last_match = matches[-1]
            sentence_end = overlap_start + last_match.end()
            return text[sentence_end:].strip()
        else:
            # Fallback: simple character-based overlap
            return text[-overlap_size:].strip()
    
    def _filter_chunks_by_size(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Filter chunks by size constraints
        
        Args:
            chunks: Input chunks
            
        Returns:
            List of filtered chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            if (self.config.min_chunk_size <= chunk.chunk_size <= self.config.max_chunk_size):
                filtered_chunks.append(chunk)
            elif chunk.chunk_size < self.config.min_chunk_size and filtered_chunks:
                # Merge with previous chunk if too small
                last_chunk = filtered_chunks[-1]
                if last_chunk.chunk_size + chunk.chunk_size <= self.config.max_chunk_size:
                    merged_chunk = self._merge_chunks(last_chunk, chunk)
                    filtered_chunks[-1] = merged_chunk
                else:
                    filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def _finalize_chunks(self, chunks: List[TextChunk], document_id: str) -> List[TextChunk]:
        """
        Finalize chunks with proper IDs and positions
        
        Args:
            chunks: Input chunks
            document_id: Document identifier
            
        Returns:
            List of finalized chunks
        """
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{document_id}_chunk_{i:04d}"
            chunk.position_in_document = i
        
        return chunks
    
    def _update_progress(self, 
                        callback: Optional[callable], 
                        stage: str, 
                        percentage: float,
                        details: str = "") -> None:
        """
        Update progress if callback provided
        
        Args:
            callback: Progress callback function
            stage: Current processing stage
            percentage: Progress percentage (0-100)
            details: Additional details
        """
        if callback and callable(callback):
            try:
                progress = ProcessingProgress(
                    document_id="current",
                    file_name=details,
                    current_stage=stage,
                    progress_percentage=percentage,
                    current_operation=stage,
                    details={"stage": stage, "details": details}
                )
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
