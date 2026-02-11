"""
PDF Text Extraction Module

Handles extraction of text content from PDF service manuals.
Supports both PyMuPDF and pdfminer backends.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PDFLoader(ABC):
    """Abstract base class for PDF loaders."""
    
    @abstractmethod
    def load(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Load and extract text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts containing page_number, text, and metadata
        """
        pass


class PyMuPDFLoader(PDFLoader):
    """PDF loader using PyMuPDF (fitz) backend."""
    
    def __init__(self, extract_tables: bool = True):
        """
        Initialize PyMuPDF loader.
        
        Args:
            extract_tables: Whether to extract table structures
        """
        self.extract_tables = extract_tables
        
    def load(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page dictionaries
        """
        import fitz  # PyMuPDF
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Loading PDF: {pdf_path}")
        
        pages = []
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                
                # Extract additional metadata
                metadata = {
                    "page_number": page_num,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                }
                
                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "metadata": metadata
                })
                
            doc.close()
            logger.info(f"Successfully loaded {len(pages)} pages")
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
            
        return pages


class PDFMinerLoader(PDFLoader):
    """PDF loader using pdfminer backend."""
    
    def load(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF using pdfminer.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page dictionaries
        """
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer, LAParams
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Loading PDF with pdfminer: {pdf_path}")
        
        pages = []
        laparams = LAParams(line_margin=0.5, word_margin=0.1)
        
        try:
            page_num = 1
            for page_layout in extract_pages(pdf_path, laparams=laparams):
                text_elements = []
                
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text_elements.append(element.get_text())
                
                text = "\n".join(text_elements)
                
                metadata = {
                    "page_number": page_num,
                    "width": page_layout.width,
                    "height": page_layout.height
                }
                
                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "metadata": metadata
                })
                
                page_num += 1
                
            logger.info(f"Successfully loaded {len(pages)} pages")
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
            
        return pages


class TextCleaner:
    """Utility class for cleaning and normalizing extracted text."""
    
    @staticmethod
    def clean_text(text: str, 
                   remove_headers_footers: bool = True,
                   normalize_whitespace: bool = True,
                   fix_line_breaks: bool = True) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            remove_headers_footers: Remove common header/footer patterns
            normalize_whitespace: Normalize spacing
            fix_line_breaks: Fix hyphenated line breaks
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Fix hyphenated line breaks
        if fix_line_breaks:
            text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
            
        # Remove excessive whitespace
        if normalize_whitespace:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
        # Remove common header/footer patterns
        if remove_headers_footers:
            # Remove page numbers at start/end of lines
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            # Remove common footer patterns
            text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
            
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, str]]:
        """
        Extract structured sections from text.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of sections with headers and content
        """
        sections = []
        
        # Pattern for section headers (all caps, or numbered sections)
        header_pattern = r'^([A-Z][A-Z\s]{2,}|\d+\.\d+\s+[A-Z].+)$'
        
        lines = text.split('\n')
        current_section = {"header": "Introduction", "content": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if re.match(header_pattern, line):
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                # Start new section
                current_section = {"header": line, "content": ""}
            else:
                current_section["content"] += line + " "
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
            
        return sections


def load_pdf(pdf_path: str, 
             loader_type: str = "pymupdf",
             clean: bool = True,
             **kwargs) -> List[Dict[str, any]]:
    """
    High-level function to load and process PDF.
    
    Args:
        pdf_path: Path to PDF file
        loader_type: Type of loader ("pymupdf" or "pdfminer")
        clean: Whether to clean extracted text
        **kwargs: Additional arguments for text cleaning
        
    Returns:
        List of processed pages
    """
    # Select loader
    if loader_type.lower() == "pymupdf":
        loader = PyMuPDFLoader()
    elif loader_type.lower() == "pdfminer":
        loader = PDFMinerLoader()
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
    
    # Load pages
    pages = loader.load(pdf_path)
    
    # Clean text if requested
    if clean:
        cleaner = TextCleaner()
        for page in pages:
            page["text"] = cleaner.clean_text(
                page["text"],
                remove_headers_footers=kwargs.get("remove_headers_footers", True),
                normalize_whitespace=kwargs.get("normalize_whitespace", True),
                fix_line_breaks=kwargs.get("fix_line_breaks", True)
            )
    
    return pages


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        pages = load_pdf(pdf_path)
        
        print(f"Loaded {len(pages)} pages")
        print(f"First page preview:\n{pages[0]['text'][:500]}...")
