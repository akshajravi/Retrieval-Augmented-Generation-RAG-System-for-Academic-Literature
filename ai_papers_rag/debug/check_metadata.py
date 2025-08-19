import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path='./data/vector_db')
collection = client.get_or_create_collection('ai_papers')

# Get sample metadata from each paper
results = collection.get(limit=20, include=['metadatas'])
metadatas = results.get('metadatas', [])

print('Sample metadata from database:')
unique_sources = set()
for metadata in metadatas:
    source = metadata.get('source_file', 'Unknown')
    if source not in unique_sources:
        unique_sources.add(source)
        print(f'\nPaper: {source}')
        print(f'  Title: {metadata.get("title", "No title")}')
        print(f'  Authors: {metadata.get("authors", "No authors")}')
        print(f'  Chunk ID: {metadata.get("chunk_id", "No chunk_id")}')
        if len(unique_sources) >= 3:  # Show first 3 papers
            break