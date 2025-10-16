"""
Context retriever for RAG system
Implements hybrid search (semantic + keyword) with ranking and consolidation
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import time

from .models import (
    DocumentChunk, RetrievedContext, RAGQuery, SearchType, 
    RetrievalStrategy, RetrievalMetrics
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

logger = logging.getLogger(__name__)

class ContextRetriever:
    """
    Retrieves relevant document chunks using hybrid search approach
    Combines semantic search and keyword search for comprehensive retrieval
    """
    
    def __init__(self, 
                 vector_store: Chroma,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 top_k_retrieval: int = 10,
                 top_k_final: int = 3):
        
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize BM25 retriever from existing documents
        self.bm25_retriever = self._initialize_bm25_retriever()
        
        logger.info("Context Retriever initialized with hybrid search")
    
    def _initialize_bm25_retriever(self) -> Optional[BM25Retriever]:
        """
        Initialize BM25 retriever from documents in vector store
        
        Returns:
            BM25Retriever or None if no documents available
        """
        try:
            # Get all documents from vector store for BM25
            # Note: This is a simplified approach - in production you'd want more efficient handling
            all_docs = self._get_all_documents_from_store()
            
            if not all_docs:
                logger.warning("No documents found in vector store for BM25 initialization")
                return None
            
            # Convert to LangChain Document format for BM25
            lc_documents = []
            for doc_chunk in all_docs:
                lc_doc = Document(
                    page_content=doc_chunk.content,
                    metadata={
                        "document_id": doc_chunk.document_id,
                        "chunk_id": doc_chunk.chunk_id,
                        "page_number": doc_chunk.page_number,
                        "section_title": doc_chunk.section_title
                    }
                )
                lc_documents.append(lc_doc)
            
            return BM25Retriever.from_documents(lc_documents)
            
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 retriever: {e}")
            return None
    
    def _get_all_documents_from_store(self) -> List[DocumentChunk]:
        """
        Extract all documents from vector store for BM25 initialization
        This is a simplified implementation - in production you'd use more efficient methods
        
        Returns:
            List of DocumentChunk objects
        """
        # This would need to be implemented based on your specific vector store setup
        # For now, return empty list - BM25 will be disabled
        return []
    
    def retrieve_context(self, query: RAGQuery) -> Tuple[RetrievedContext, RetrievalMetrics]:
        """
        Retrieve relevant context for a query using hybrid search
        
        Args:
            query: Enhanced RAG query object
            
        Returns:
            Tuple of (RetrievedContext, RetrievalMetrics)
        """
        start_time = time.time()
        logger.info(f"Retrieving context for query: '{query.original_query}'")
        
        retrieval_results = []
        search_types_used = []
        all_chunks = []
        all_scores = []
        
        # Perform retrieval based on strategy
        if query.retrieval_strategy == RetrievalStrategy.MULTI_QUERY:
            retrieval_results = self._multi_query_retrieval(query)
        elif query.retrieval_strategy == RetrievalStrategy.HYBRID_RERANK:
            retrieval_results = self._hybrid_rerank_retrieval(query)
        else:  # SINGLE_SHOT
            retrieval_results = self._single_shot_retrieval(query)
        
        # Extract chunks and scores from results
        for result in retrieval_results:
            all_chunks.extend(result["chunks"])
            all_scores.extend(result["scores"])
            search_types_used.append(result["search_type"])
        
        # Remove duplicates and consolidate
        consolidated_chunks, consolidated_scores = self._consolidate_results(all_chunks, all_scores)
        
        # Apply similarity threshold
        filtered_chunks, filtered_scores = self._filter_by_threshold(
            consolidated_chunks, consolidated_scores
        )
        
        # Get top K chunks
        final_chunks, final_scores = self._get_top_k_chunks(filtered_chunks, filtered_scores)
        
        # Calculate metrics
        retrieval_time = time.time() - start_time
        metrics = RetrievalMetrics(
            retrieval_time=retrieval_time,
            chunks_considered=len(consolidated_chunks),
            chunks_retrieved=len(final_chunks),
            avg_similarity_score=sum(final_scores) / len(final_scores) if final_scores else 0.0,
            search_types_used=search_types_used,
            query_expansion_count=len(query.enhanced_queries)
        )
        
        # Create retrieved context
        context = RetrievedContext(
            chunks=final_chunks,
            search_scores=final_scores,
            search_type=query.search_type,
            retrieval_strategy=query.retrieval_strategy
        )
        
        logger.info(f"Retrieved {len(final_chunks)} chunks in {retrieval_time:.2f}s")
        return context, metrics
    
    def _single_shot_retrieval(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """
        Perform single-shot retrieval using the original query
        
        Args:
            query: RAG query object
            
        Returns:
            List of retrieval results
        """
        results = []
        
        # Use the first enhanced query (usually the original)
        search_query = query.enhanced_queries[0] if query.enhanced_queries else query.original_query
        
        if query.search_type in [SearchType.SEMANTIC, SearchType.HYBRID]:
            semantic_results = self._semantic_search(search_query)
            results.append(semantic_results)
        
        if query.search_type in [SearchType.KEYWORD, SearchType.HYBRID] and self.bm25_retriever:
            keyword_results = self._keyword_search(search_query)
            results.append(keyword_results)
        
        return results
    
    def _multi_query_retrieval(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """
        Perform retrieval using multiple query variations
        
        Args:
            query: RAG query object with enhanced queries
            
        Returns:
            List of retrieval results
        """
        results = []
        
        for enhanced_query in query.enhanced_queries:
            if query.search_type in [SearchType.SEMANTIC, SearchType.HYBRID]:
                semantic_results = self._semantic_search(enhanced_query)
                results.append(semantic_results)
            
            if query.search_type in [SearchType.KEYWORD, SearchType.HYBRID] and self.bm25_retriever:
                keyword_results = self._keyword_search(enhanced_query)
                results.append(keyword_results)
        
        return results
    
    def _hybrid_rerank_retrieval(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with reranking
        
        Args:
            query: RAG query object
            
        Returns:
            List of retrieval results
        """
        # First, get results from both methods
        semantic_results = []
        keyword_results = []
        
        for enhanced_query in query.enhanced_queries:
            if query.search_type in [SearchType.SEMANTIC, SearchType.HYBRID]:
                semantic_results.extend(self._semantic_search(enhanced_query)["chunks"])
            
            if query.search_type in [SearchType.KEYWORD, SearchType.HYBRID] and self.bm25_retriever:
                keyword_results.extend(self._keyword_search(enhanced_query)["chunks"])
        
        # Combine and rerank
        all_chunks = semantic_results + keyword_results
        
        # Remove duplicates
        unique_chunks = []
        seen_ids = set()
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                unique_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)
        
        # Rerank using combined scoring (simplified approach)
        reranked_chunks = self._rerank_chunks(unique_chunks, query.original_query)
        
        return [{
            "chunks": reranked_chunks,
            "scores": [1.0] * len(reranked_chunks),  # Placeholder scores
            "search_type": SearchType.HYBRID
        }]
    
    def _semantic_search(self, query: str) -> Dict[str, Any]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query: Search query
            
        Returns:
            Dict with chunks and scores
        """
        try:
            # This would use your actual vector store implementation
            # For now, return empty results as placeholder
            return {
                "chunks": [],
                "scores": [],
                "search_type": SearchType.SEMANTIC
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                "chunks": [],
                "scores": [], 
                "search_type": SearchType.SEMANTIC
            }
    
    def _keyword_search(self, query: str) -> Dict[str, Any]:
        """
        Perform keyword search using BM25
        
        Args:
            query: Search query
            
        Returns:
            Dict with chunks and scores
        """
        if not self.bm25_retriever:
            return {
                "chunks": [],
                "scores": [],
                "search_type": SearchType.KEYWORD
            }
        
        try:
            # This would use your actual BM25 implementation
            # For now, return empty results as placeholder
            return {
                "chunks": [],
                "scores": [],
                "search_type": SearchType.KEYWORD
            }
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return {
                "chunks": [],
                "scores": [],
                "search_type": SearchType.KEYWORD
            }
    
    def _rerank_chunks(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """
        Rerank chunks based on relevance to query
        
        Args:
            chunks: List of document chunks
            query: Original query
            
        Returns:
            Reranked list of chunks
        """
        # Simple reranking based on keyword matching
        # In production, you might use a cross-encoder or more sophisticated approach
        
        scored_chunks = []
        query_terms = set(query.lower().split())
        
        for chunk in chunks:
            score = 0
            chunk_terms = set(chunk.content.lower().split())
            
            # Basic term overlap scoring
            overlap = query_terms.intersection(chunk_terms)
            score += len(overlap) * 0.1
            
            # Position scoring - chunks with query terms early get higher scores
            for i, term in enumerate(query_terms):
                if term in chunk.content.lower():
                    position = chunk.content.lower().find(term)
                    if position != -1:
                        score += 1.0 / (position + 1)  # Higher score for earlier occurrence
            
            scored_chunks.append((chunk, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks]
    
    def _consolidate_results(self, chunks: List[DocumentChunk], scores: List[float]) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Remove duplicate chunks and consolidate scores
        
        Args:
            chunks: List of document chunks
            scores: Corresponding similarity scores
            
        Returns:
            Tuple of (consolidated chunks, consolidated scores)
        """
        unique_chunks = []
        unique_scores = []
        seen_chunk_ids = set()
        
        for chunk, score in zip(chunks, scores):
            if chunk.chunk_id not in seen_chunk_ids:
                unique_chunks.append(chunk)
                unique_scores.append(score)
                seen_chunk_ids.add(chunk.chunk_id)
        
        return unique_chunks, unique_scores
    
    def _filter_by_threshold(self, chunks: List[DocumentChunk], scores: List[float]) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Filter chunks by similarity threshold
        
        Args:
            chunks: List of document chunks
            scores: Corresponding similarity scores
            
        Returns:
            Tuple of (filtered chunks, filtered scores)
        """
        filtered_chunks = []
        filtered_scores = []
        
        for chunk, score in zip(chunks, scores):
            if score >= self.similarity_threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)
        
        logger.debug(f"Filtered {len(chunks)} -> {len(filtered_chunks)} chunks by threshold {self.similarity_threshold}")
        return filtered_chunks, filtered_scores
    
    def _get_top_k_chunks(self, chunks: List[DocumentChunk], scores: List[float]) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Get top K chunks by score
        
        Args:
            chunks: List of document chunks
            scores: Corresponding similarity scores
            
        Returns:
            Tuple of (top K chunks, top K scores)
        """
        if len(chunks) <= self.top_k_final:
            return chunks, scores
        
        # Sort by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_chunks = [chunks[i] for i in sorted_indices[:self.top_k_final]]
        top_scores = [scores[i] for i in sorted_indices[:self.top_k_final]]
        
        return top_chunks, top_scores
    
    def validate_retrieval_quality(self, context: RetrievedContext, query: str) -> bool:
        """
        Basic validation of retrieval quality
        
        Args:
            context: Retrieved context
            query: Original query
            
        Returns:
            bool: True if retrieval quality is acceptable
        """
        if not context.chunks:
            logger.warning("No chunks retrieved for query")
            return False
        
        if context.average_score < self.similarity_threshold:
            logger.warning(f"Low average similarity score: {context.average_score}")
            return False
        
        # Check if any chunks contain query terms
        query_terms = set(query.lower().split())
        relevant_chunks = 0
        
        for chunk in context.chunks:
            chunk_terms = set(chunk.content.lower().split())
            if query_terms.intersection(chunk_terms):
                relevant_chunks += 1
        
        relevance_ratio = relevant_chunks / len(context.chunks)
        if relevance_ratio < 0.5:  # At least 50% of chunks should be relevant
            logger.warning(f"Low relevance ratio: {relevance_ratio}")
            return False
        
        return True
