from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
import logging
import time

class EmbeddingService(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.logger  = logging.getLogger(__name__)

        self.model_dimensions = {
            "text-embedding-ada-002" : 1536,
            "text-embedding-3-small" : 1536,
            "text-embedding-3-large" : 3072
        }

        self.last_request_time = 0
        self.min_request_interval = .1
    
    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        self._rate_limit()

        try:
            response = self.client.embeddings.create(
                input=text.strip(),
                model=self.model
            )

            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Open AI embedding error for text: {str(e)}")
            raise
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        filtered_texts = []
        text_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                filtered_texts.append(text.strip())
                text_indices.append(i)
        if not filtered_texts:
            return [[0.0] * self.get_embedding_dimension()] * len(texts)
        
        self._rate_limit()

        try:
            batch_size = 100
            all_embeddings = []

            for i in range(0,len(filtered_texts),batch_size):
                batch = filtered_texts[i:i + batch_size]

                self.logger.info(f"Processing embedding batch {i//batch_size + 1}, size: {len(batch)}")

                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                if i + batch_size < len(filtered_texts):
                    time.sleep(0.1)
            result = [[0.0] * self.get_embedding_dimension()] * len(texts)
            for i, embedding in enumerate(all_embeddings):
                original_index = text_indices[i]
                result[original_index] = embedding
            return result
        except Exception as e:
            self.logger.error(f"OpenAI batch embedding error: {str(e)}")
            raise
    def get_embedding_dimension(self) -> int:
        return self.model_dimensions.get(self.model,1536)
    

    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

class HuggingFaceEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded HuggingFace model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model {model_name}: {str(e)}")
            raise

    
    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embedding = self.model.encode(text.strip(),convert_to_tensor = False)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"HuggingFace embedding: {str(e)}")
            raise


    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts efficiently."""
        if not texts:
            return []
        
        # Filter empty texts
        filtered_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not filtered_texts:
            return [[0.0] * self.get_embedding_dimension()] * len(texts)
        
        try:
            self.logger.info(f"Processing batch of {len(filtered_texts)} texts with HuggingFace")
            
            # HuggingFace can handle large batches efficiently
            embeddings = self.model.encode(
                filtered_texts,
                convert_to_tensor=False,
                show_progress_bar=True if len(filtered_texts) > 100 else False,
                batch_size=32  
            )
            
            result = []
            filtered_idx = 0
            
            for original_text in texts:
                if original_text and original_text.strip():
                    result.append(embeddings[filtered_idx].tolist())
                    filtered_idx += 1
                else:
                    result.append([0.0] * self.get_embedding_dimension())
            
            return result
        
        except Exception as e:
            self.logger.error(f"HuggingFace batch embedding error: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        try:
            sample_embedding = self.model.encode("sample text", convert_to_tensor=False)
            return len(sample_embedding)
        except Exception as e:
            self.logger.error(f"Could not determine embedding dimension: {str(e)}")
            dimension_map = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-MiniLM-L12-v2": 384,
            }
            return dimension_map.get(self.model_name, 384)

def create_embedding_service(provider: str = "openai", **kwargs) -> EmbeddingService:
    """Factory function to create embedding service instances."""
    if provider.lower() == "openai":
        return OpenAIEmbeddingService(**kwargs)
    elif provider.lower() == "huggingface":
        return HuggingFaceEmbeddingService(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")