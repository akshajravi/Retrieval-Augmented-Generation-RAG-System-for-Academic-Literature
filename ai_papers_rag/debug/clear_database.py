#!/usr/bin/env python3

import chromadb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_database():
    try:
        # Get database path from config
        db_path = os.getenv('VECTOR_DB_PATH', './data/vector_db')
        print(f"Clearing database at: {db_path}")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=db_path)
        
        # Check current state
        try:
            collection = client.get_collection('ai_papers')
            count = collection.count()
            print(f"Current documents in database: {count}")
        except:
            print("Collection 'ai_papers' does not exist yet")
            return
        
        # Delete the collection
        client.delete_collection('ai_papers')
        print("✅ Database cleared successfully!")
        print("You can now run the enhanced ingestion script.")
        
    except Exception as e:
        print(f"❌ Error clearing database: {e}")

if __name__ == "__main__":
    print("🗑️ Database Cleaner")
    print("=" * 30)
    
    response = input("Are you sure you want to clear the database? (y/N): ")
    if response.lower() in ['y', 'yes']:
        clear_database()
    else:
        print("Database clearing cancelled.")