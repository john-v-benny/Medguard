"""
Vector Database Setup for Bayesian Medical Chatbot
Creates ChromaDB vector database from disease knowledge base.
"""

import csv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os


def load_knowledge_base(csv_file: str = 'disease_knowledge_base.csv') -> List[Dict]:
    """Load disease knowledge base from CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Knowledge base file '{csv_file}' not found!\n"
            f"Please run: python create_knowledge_base.py first"
        )
    
    diseases = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases.append(row)
    
    return diseases


def create_vector_database(
    knowledge_base: List[Dict],
    db_path: str = './chroma_db',
    collection_name: str = 'disease_knowledge'
):
    """
    Create ChromaDB vector database from knowledge base.
    
    Args:
        knowledge_base: List of disease dictionaries
        db_path: Path to store ChromaDB
        collection_name: Name of the collection
    """
    print(f"\n{'='*80}")
    print("CREATING VECTOR DATABASE")
    print(f"{'='*80}\n")
    
    # Initialize embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Medical disease knowledge base from MedlinePlus"}
    )
    print(f"✓ Created collection: {collection_name}\n")
    
    # Prepare documents and metadata
    print(f"Processing {len(knowledge_base)} diseases...")
    
    documents = []
    metadatas = []
    ids = []
    
    for i, disease_data in enumerate(knowledge_base):
        # Create comprehensive document for embedding
        # Combine multiple fields for better semantic search
        doc = f"""
        Disease: {disease_data['disease']}
        
        Description: {disease_data['description']}
        
        Symptoms: {disease_data['symptoms']}
        
        Causes: {disease_data['causes']}
        
        Treatment: {disease_data['treatment']}
        
        Medications: {disease_data['medications']}
        
        Prevention: {disease_data['prevention']}
        """.strip()
        
        documents.append(doc)
        
        # Store all metadata
        metadatas.append({
            'disease': disease_data['disease'],
            'description': disease_data['description'][:500],  # Truncate for metadata
            'symptoms': disease_data['symptoms'][:500],
            'causes': disease_data['causes'][:500],
            'treatment': disease_data['treatment'][:1000],
            'medications': disease_data['medications'][:500],
            'diagnosis': disease_data['diagnosis'][:500],
            'prevention': disease_data['prevention'][:500],
            'complications': disease_data['complications'][:500],
            'when_to_see_doctor': disease_data['when_to_see_doctor'][:500],
            'source_url': disease_data['source_url'],
            'source_name': disease_data['source_name'],
            'last_updated': disease_data['last_updated']
        })
        
        ids.append(f"disease_{i}")
        
        print(f"  [{i+1}/{len(knowledge_base)}] Processed: {disease_data['disease']}")
    
    # Generate embeddings
    print(f"\nGenerating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print("✓ Embeddings generated\n")
    
    # Add to collection
    print("Adding documents to ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    print("✓ Documents added to collection\n")
    
    # Verify
    count = collection.count()
    print(f"{'='*80}")
    print(f"VECTOR DATABASE CREATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"  Database path: {db_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Total documents: {count}")
    print(f"  Embedding model: all-MiniLM-L6-v2")
    print(f"  Embedding dimension: {len(embeddings[0])}")
    print(f"{'='*80}\n")


def test_database(db_path: str = './chroma_db', collection_name: str = 'disease_knowledge'):
    """Test the vector database with sample queries."""
    print(f"\n{'='*80}")
    print("TESTING VECTOR DATABASE")
    print(f"{'='*80}\n")
    
    # Load collection
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test queries
    test_queries = [
        "diabetes high blood sugar",
        "fever headache malaria",
        "chest pain heart attack"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 80)
        
        # Generate query embedding
        query_embedding = model.encode([query])[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        # Display results
        for i, (disease, distance) in enumerate(zip(
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"  {i}. {disease['disease']} (similarity: {1-distance:.3f})")
            print(f"     {disease['description'][:100]}...")
        
        print()


def main():
    """Main function to create vector database."""
    print("\n" + "="*80)
    print("BAYESIAN MEDICAL CHATBOT - VECTOR DATABASE SETUP")
    print("="*80)
    
    # Check if knowledge base exists
    kb_file = 'disease_knowledge_base.csv'
    if not os.path.exists(kb_file):
        print(f"\n❌ Error: {kb_file} not found!")
        print("\nPlease run the knowledge base creator first:")
        print("  python create_knowledge_base.py")
        return
    
    # Load knowledge base
    print(f"\nLoading knowledge base from: {kb_file}")
    knowledge_base = load_knowledge_base(kb_file)
    print(f"✓ Loaded {len(knowledge_base)} diseases\n")
    
    # Create vector database
    create_vector_database(knowledge_base)
    
    # Test database
    test_database()
    
    print("="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Test RAG retrieval: python test_rag.py")
    print("2. Integrate with chatbot: python hybrid_chatbot.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
