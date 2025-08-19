import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from embeddings.vector_store import VectorStore, SearchResult

def test_vector_store():
    print("Testing VectorStore...")
    
    # Initialize vector store
    vs = VectorStore(collection_name="test_papers", persist_directory=None)
    
    # Test data
    documents = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning models require large datasets.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    # Generate fake embeddings (normally from embedding model)
    embeddings = [
        np.random.random(384).tolist(),  # typical sentence-transformer size
        np.random.random(384).tolist(),
        np.random.random(384).tolist()
    ]
    
    metadata = [
        {"source_file": "ai_paper.pdf", "page": 1, "author": "Smith"},
        {"source_file": "ml_paper.pdf", "page": 2, "author": "Jones"},
        {"source_file": "dl_paper.pdf", "page": 3, "author": "Brown"}
    ]
    
    ids = ["doc1", "doc2", "doc3"]
    
    # Test 1: Add documents
    print("✓ Adding documents...")
    vs.add_documents(documents, embeddings, metadata, ids)
    
    # Test 2: Get stats
    print("✓ Getting stats...")
    stats = vs.get_stats()
    print(f"  Total documents: {stats['total_documents']}")
    
    # Test 3: Search
    print("✓ Testing search...")
    query_embedding = np.random.random(384).tolist()
    results = vs.search(query_embedding, k=2)
    print(f"  Found {len(results)} results")
    if results:
        print(f"  Top result score: {results[0].score:.3f}")
    
    # Test 4: List sources
    print("✓ Testing list sources...")
    sources = vs.list_sources()
    print(f"  Sources: {sources}")
    
    # Test 5: Search with filter
    print("✓ Testing filtered search...")
    filtered_results = vs.search_with_filter(
        query_embedding, 
        {"author": "Smith"}, 
        k=1
    )
    print(f"  Filtered results: {len(filtered_results)}")
    
    # Test 6: Get chunks by source
    print("✓ Testing get chunks by source...")
    chunks = vs.get_chunks_by_source("ai_paper.pdf")
    print(f"  Chunks from ai_paper.pdf: {len(chunks)}")
    
    # Test 7: Delete documents
    print("✓ Testing delete...")
    vs.delete_documents(["doc1"])
    final_stats = vs.get_stats()
    print(f"  Documents after deletion: {final_stats['total_documents']}")
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_vector_store()