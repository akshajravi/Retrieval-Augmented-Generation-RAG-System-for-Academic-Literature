import chromadb
import os
from dotenv import load_dotenv
import openai

load_dotenv()

# Test the retrieval pipeline directly
client = chromadb.PersistentClient(path='./data/vector_db')
collection = client.get_or_create_collection('ai_papers')

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Test query
test_query = "What are attention mechanisms?"

print(f"Testing query: '{test_query}'")
print("=" * 50)

# Get embedding for query
try:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=test_query
    )
    query_embedding = response.data[0].embedding
    print(f"✓ Got query embedding (dimension: {len(query_embedding)})")
except Exception as e:
    print(f"❌ Error getting embedding: {e}")
    exit()

# Search ChromaDB directly
try:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"✓ ChromaDB returned {len(results['documents'][0])} results")
    print("\nDirect ChromaDB results:")
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0] 
    distances = results['distances'][0]
    ids = results['ids'][0]
    
    for i in range(len(documents)):
        similarity_score = 1.0 - distances[i] if distances[i] <= 1.0 else 0.0
        
        print(f"\nResult {i+1}:")
        print(f"  Distance: {distances[i]}")
        print(f"  Similarity: {similarity_score}")
        print(f"  Title: {metadatas[i].get('title', 'Unknown')}")
        print(f"  Authors: {metadatas[i].get('authors', 'Unknown')}")
        print(f"  Content preview: {documents[i][:100]}...")
        
except Exception as e:
    print(f"❌ Error querying ChromaDB: {e}")