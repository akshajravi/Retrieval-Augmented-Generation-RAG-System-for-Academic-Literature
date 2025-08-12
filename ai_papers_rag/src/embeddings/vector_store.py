from typing import List, Dict, Any, Optional, Tuple
import chromadb
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str


class ChromaVectorStore:
    def __init__(self, collection_name: str = "papers", persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)

        try:
            if persist_directory:

                Path(persist_directory).mkdir(parents = True, exist_ok = True)
                self.client = chromadb.PersistentClient(path = persist_directory)
            else:
                self.client = chromadb.Client()

            self.collection = self.client.get_or_create_collection(
                name = collection_name,
                metadata = {"description" : "Academic papers collection"}
            )

            self.logger.info(f"ChromaDB collection '{collection_name}' initialized")

        except Exception as e:
            self.logger.error(f"failed to initialize ChromaDB : {str(e)}")
            raise



    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]], ids: List[str]) -> None:
        lengths = [len(documents), len(embeddings),len(metadata),len(ids)]

        if not documents or len(set(lengths)) !=1:
            raise ValueError("All input lists must have the same lengths and be non empty man")

        try:
            embeddings_array = np.array(embeddings).astype(np.float32).tolist()

            clean_metadata = []
            for meta in metadata:
                clean_meta = {}
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool, list)):
                        clean_meta[key] = value
                    else:
                        clean_meta[key] = str(value)
                clean_metadata.append(clean_meta)

            self.collection.add(
                documents = documents,
                embeddings = embeddings_array,
                metadatas = clean_metadata,
                ids = ids
            )

            self.logger.info(f"Added {len(documents)} documents to ChromaDB")

        except Exception as e:
            self.logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        try:
            query_array = np.array(query_embedding).astype(np.float32).tolist()

            results = self.collection.query(
                query_embeddings = [query_array],
                n_results = k,
                include = ["documents","metadatas","distances"]
            )

            search_results = []
            
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            ids = results['ids'][0] if results['ids'] else []
            
            for i in range(len(documents)):

                similarity_score = 1.0 - distances[i] if distances[i] <= 1.0 else 0.0
                
                search_results.append(SearchResult(
                    content=documents[i],
                    metadata=metadatas[i] if metadatas else {},
                    score=similarity_score,
                    chunk_id=ids[i]
                ))
            
            self.logger.info(f"Found {len(search_results)} results for query")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error searching ChromaDB: {str(e)}")
            return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            self.logger.error(f"Error deleting documents from ChromaDB: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error getting ChromaDB stats: {str(e)}")
            return {"error": str(e)}
    
    def search_with_filter(self, query_embedding: List[float], filters: Dict[str, Any], 
                          k: int = 5) -> List[SearchResult]:
        """Search with metadata filters (e.g., specific paper, author, etc.)."""
        try:
            query_array = np.array(query_embedding).astype(np.float32).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_array],
                n_results=k,
                where=filters,  # ChromaDB metadata filtering
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            ids = results['ids'][0] if results['ids'] else []
            
            for i in range(len(documents)):
                similarity_score = 1.0 - distances[i] if distances[i] <= 1.0 else 0.0
                
                search_results.append(SearchResult(
                    content=documents[i],
                    metadata=metadatas[i] if metadatas else {},
                    score=similarity_score,
                    chunk_id=ids[i]
                ))
            
            self.logger.info(f"Found {len(search_results)} filtered results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in filtered search: {str(e)}")
            return []
    
    def delete_by_source(self, source_file: str) -> None:
        """Delete all documents from a specific source file."""
        try:
            self.collection.delete(where={"source_file": source_file})
            self.logger.info(f"Deleted all documents from {source_file}")
        except Exception as e:
            self.logger.error(f"Error deleting documents from {source_file}: {str(e)}")
            raise
    
    def list_sources(self) -> List[str]:
        """Get list of all source files in the collection."""
        try:
            # Get all documents to extract unique sources
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            sources = set()
            for metadata in metadatas:
                if "source_file" in metadata:
                    sources.add(metadata["source_file"])
            
            return sorted(list(sources))
            
        except Exception as e:
            self.logger.error(f"Error listing sources: {str(e)}")
            return []
    
    def get_chunks_by_source(self, source_file: str) -> List[SearchResult]:
        """Get all chunks from a specific source file."""
        try:
            results = self.collection.get(
                where={"source_file": source_file},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])
            
            for i in range(len(documents)):
                chunks.append(SearchResult(
                    content=documents[i],
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    score=1.0,  # No similarity score for direct retrieval
                    chunk_id=ids[i] if i < len(ids) else f"unknown_{i}"
                ))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error getting chunks from {source_file}: {str(e)}")
            return []