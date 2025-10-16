"""
PDF Processor for text extraction using pdfplumber
Handles PDF text extraction with error handling and progress tracking
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import hashlib

import pdfplumber
from PIL import Image
import magic

from .models import (
    DocumentMetadata, PageText, ExtractedDocument,
    ProcessingStatus, DocumentType, ProcessingProgress
)

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Processes PDF files for text extraction with robust error handling
    and comprehensive metadata extraction
    """
    
    def __init__(self, 
                 enable_image_processing: bool = False,
                 max_pages: Optional[int] = None,
                 timeout_seconds: int = 300):
        """
        Initialize PDF Processor
        
        Args:
            enable_image_processing: Whether to attempt OCR on image-based PDFs
            max_pages: Maximum number of pages to process (None for all)
            timeout_seconds: Maximum processing time per PDF
        """
        self.enable_image_processing = enable_image_processing
        self.max_pages = max_pages
        self.timeout_seconds = timeout_seconds
        
        # Supported PDF MIME types
        self.supported_mime_types = [
            'application/pdf',
            'application/x-pdf'
        ]
        
        logger.info("PDF Processor initialized")
    
    def extract_text(self, 
                    file_path: Path,
                    progress_callback: Optional[callable] = None) -> ExtractedDocument:
        """
        Extract text from PDF file with comprehensive error handling
        
        Args:
            file_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            ExtractedDocument: Extracted text and metadata
            
        Raises:
            ValueError: If file is not a valid PDF
            IOError: If file cannot be read
            TimeoutError: If processing takes too long
        """
        start_time = time.time()
        logger.info(f"Starting PDF extraction: {file_path}")
        
        # Validate file
        self._validate_file(file_path)
        
        # Create document metadata
        metadata = self._extract_basic_metadata(file_path)
        
        try:
            # Update progress
            self._update_progress(progress_callback, "validating", 10, file_path.name)
            
            # Extract text from PDF
            pages = self._extract_pages(file_path, progress_callback)
            
            # Update progress
            self._update_progress(progress_callback, "extracting_metadata", 90, file_path.name)
            
            # Extract additional metadata from content
            enhanced_metadata = self._enhance_metadata(metadata, pages)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ExtractedDocument(
                metadata=enhanced_metadata,
                pages=pages,
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED
            )
            
            self._update_progress(progress_callback, "completed", 100, file_path.name)
            logger.info(f"PDF extraction completed: {file_path} - {len(pages)} pages, {result.total_word_count} words")
            
            return result
            
        except pdfplumber.PDFSyntaxError as e:
            error_msg = f"Invalid PDF structure: {e}"
            logger.error(f"PDF syntax error for {file_path}: {e}")
            return self._create_error_result(metadata, error_msg, start_time)
            
        except Exception as e:
            error_msg = f"Unexpected error during PDF processing: {e}"
            logger.error(f"PDF processing failed for {file_path}: {e}")
            return self._create_error_result(metadata, error_msg, start_time)
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate PDF file before processing
        
        Args:
            file_path: Path to PDF file
            
        Raises:
            ValueError: If file is invalid
            IOError: If file cannot be accessed
        """
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {file_path}")
        
        # Check file size limit (100MB)
        if file_size > 100 * 1024 * 1024:
            raise ValueError(f"PDF file too large: {file_size} bytes (max 100MB)")
        
        # Check MIME type
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))
            
            if mime_type not in self.supported_mime_types:
                raise ValueError(f"Unsupported file type: {mime_type}. Expected PDF.")
                
        except Exception as e:
            logger.warning(f"Could not verify MIME type for {file_path}: {e}")
            # Continue with processing if MIME check fails
    
    def _extract_basic_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Extract basic metadata from file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            DocumentMetadata: Basic file metadata
        """
        file_stat = file_path.stat()
        
        return DocumentMetadata(
            file_path=file_path,
            file_name=file_path.name,
            file_size=file_stat.st_size,
            document_type=DocumentType.PDF,
            modification_date=datetime.fromtimestamp(file_stat.st_mtime)
        )
    
    def _extract_pages(self, 
                      file_path: Path, 
                      progress_callback: Optional[callable] = None) -> List[PageText]:
        """
        Extract text from all pages in PDF
        
        Args:
            file_path: Path to PDF file
            progress_callback: Optional progress callback
            
        Returns:
            List of PageText objects
        """
        pages = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = self.max_pages or total_pages
                
                logger.info(f"Processing {pages_to_process} pages from {total_pages} total")
                
                for page_num, pdf_page in enumerate(pdf.pages[:pages_to_process], 1):
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        raise TimeoutError(f"PDF processing timeout after {self.timeout_seconds} seconds")
                    
                    # Update progress
                    progress = 10 + (page_num / pages_to_process) * 80  # 10-90%
                    self._update_progress(progress_callback, "extracting_pages", progress, 
                                        f"Page {page_num}/{pages_to_process}")
                    
                    # Extract page text
                    page_text = self._extract_single_page(pdf_page, page_num)
                    pages.append(page_text)
                    
                    logger.debug(f"Extracted page {page_num}: {page_text.word_count} words")
        
        except pdfplumber.PDFSyntaxError:
            raise  # Re-raise syntax errors
        except Exception as e:
            logger.error(f"Error during page extraction for {file_path}: {e}")
            raise
        
        return pages
    
    def _extract_single_page(self, pdf_page, page_num: int) -> PageText:
        """
        Extract text from a single PDF page
        
        Args:
            pdf_page: pdfplumber page object
            page_num: Page number
            
        Returns:
            PageText: Extracted page text with metadata
        """
        try:
            # Extract text with layout preservation
            text = pdf_page.extract_text(
                layout=True,  # Preserve layout
                x_tolerance=3,  # Horizontal tolerance
                y_tolerance=3   # Vertical tolerance
            ) or ""  # Handle None returns
            
            # Clean up text
            cleaned_text = self._clean_extracted_text(text)
            
            # Count images and tables
            images_count = len(pdf_page.images) if hasattr(pdf_page, 'images') else 0
            tables_count = len(pdf_page.find_tables()) if hasattr(pdf_page, 'find_tables') else 0
            
            # Extract sections (simple heuristic)
            sections = self._extract_sections_from_page(cleaned_text)
            
            return PageText(
                page_number=page_num,
                text=cleaned_text,
                sections=sections,
                images_count=images_count,
                tables_count=tables_count
            )
            
        except Exception as e:
            logger.warning(f"Error extracting page {page_num}: {e}")
            # Return empty page text on error
            return PageText(
                page_number=page_num,
                text="",
                sections=[],
                images_count=0,
                tables_count=0
            )
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common PDF extraction artifacts
        replacements = [
            (r'\s+\.\s+', '. '),  # Spaces around periods
            (r'\s+,\s+', ', '),   # Spaces around commas
            (r'\s+;\s+', '; '),   # Spaces around semicolons
            (r'\s+:\s+', ': '),   # Spaces around colons
            (r'\s+-\s+', '-'),    # Spaces around hyphens
            (r'\uf0b7', '•'),     # Replace bullet characters
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        # Ensure proper paragraph separation
        text = re.sub(r'\.\s*([A-Z])', r'. \n\n\1', text)  # New paragraph after sentences
        
        return text.strip()
    
    def _extract_sections_from_page(self, text: str) -> List[str]:
        """
        Extract potential section headers from page text
        
        Args:
            text: Page text
            
        Returns:
            List of potential section titles
        """
        sections = []
        
        # Simple heuristic: lines that are short and end without period
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if (len(line) < 100 and  # Reasonable section header length
                len(line.split()) >= 2 and  # At least 2 words
                not line.endswith('.') and  # Not a complete sentence
                line.isprintable() and
                not line.islower()):  # Not all lowercase
                sections.append(line)
        
        return sections[:5]  # Return top 5 potential sections
    
    def _enhance_metadata(self, 
                         metadata: DocumentMetadata, 
                         pages: List[PageText]) -> DocumentMetadata:
        """
        Enhance metadata with information from PDF content
        
        Args:
            metadata: Basic metadata
            pages: Extracted pages
            
        Returns:
            Enhanced DocumentMetadata
        """
        enhanced_metadata = metadata
        enhanced_metadata.page_count = len(pages)
        
        # Extract potential title from first page
        if pages and pages[0].text:
            first_page_text = pages[0].text
            potential_title = self._extract_potential_title(first_page_text)
            if potential_title:
                enhanced_metadata.title = potential_title
        
        # Detect language from text sample
        if pages:
            sample_text = ' '.join(page.text for page in pages[:3])  # First 3 pages
            language = self._detect_language(sample_text)
            enhanced_metadata.language = language
        
        return enhanced_metadata
    
    def _extract_potential_title(self, first_page_text: str) -> Optional[str]:
        """
        Extract potential document title from first page
        
        Args:
            first_page_text: Text from first page
            
        Returns:
            Potential title or None
        """
        lines = first_page_text.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if (len(line) > 10 and  # Reasonable title length
                len(line) < 200 and
                len(line.split()) >= 2 and  # At least 2 words
                not line.endswith('.') and  # Not a sentence
                any(c.isupper() for c in line)):  # Has uppercase
                return line
        
        return None
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words
        
        Args:
            text: Text sample
            
        Returns:
            Detected language code
        """
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # Simple word frequency approach
        english_words = {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'for', 'it'}
        spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'}
        
        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        
        if english_count > spanish_count and english_count >= 2:
            return "en"
        elif spanish_count > english_count and spanish_count >= 2:
            return "es"
        else:
            return "unknown"
    
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
    
    def _create_error_result(self, 
                           metadata: DocumentMetadata, 
                           error_message: str,
                           start_time: float) -> ExtractedDocument:
        """
        Create error result for failed extraction
        
        Args:
            metadata: Document metadata
            error_message: Error description
            start_time: Processing start time
            
        Returns:
            ExtractedDocument with error status
        """
        processing_time = time.time() - start_time
        
        return ExtractedDocument(
            metadata=metadata,
            pages=[],
            processing_time=processing_time,
            status=ProcessingStatus.FAILED,
            error_message=error_message
        )
    
    def batch_process(self, 
                     file_paths: List[Path],
                     progress_callback: Optional[callable] = None) -> List[ExtractedDocument]:
        """
        Process multiple PDF files
        
        Args:
            file_paths: List of PDF file paths
            progress_callback: Optional progress callback
            
        Returns:
            List of extraction results
        """
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                # Update batch progress
                if progress_callback:
                    batch_progress = ProcessingProgress(
                        document_id=f"batch_{i}",
                        file_name=file_path.name,
                        current_stage="batch_processing",
                        progress_percentage=(i / total_files) * 100,
                        current_operation=f"Processing {i}/{total_files}",
                        details={"current_file": file_path.name, "total_files": total_files}
                    )
                    progress_callback(batch_progress)
                
                # Process individual file
                result = self.extract_text(file_path)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch processing failed for {file_path}: {e}")
                # Create error result for failed file
                metadata = self._extract_basic_metadata(file_path)
                error_result = self._create_error_result(metadata, str(e), time.time())
                results.append(error_result)
        
        return results
