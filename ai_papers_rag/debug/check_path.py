import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

print('VECTOR_DB_PATH from .env:', os.getenv('VECTOR_DB_PATH'))
print('Current working directory:', os.getcwd())

# Check if database exists at the configured path
db_path = os.getenv('VECTOR_DB_PATH', './data/vector_db')
print(f'Looking for database at: {db_path}')
print(f'Absolute path: {os.path.abspath(db_path)}')
print(f'Database directory exists: {os.path.exists(db_path)}')

if os.path.exists(db_path):
    print(f'Contents of {db_path}:')
    for item in os.listdir(db_path):
        print(f'  - {item}')

# Try to connect to database
try:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection('ai_papers')
    count = collection.count()
    print(f'Documents in database: {count}')
except Exception as e:
    print(f'Error connecting to database: {e}')