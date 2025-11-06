import json
import os
from datetime import datetime
import time
from tqdm import tqdm
import argparse

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document as LangchainDocument

# ===== CONFIGURATION =====
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # generic model (fast)
# MODEL_NAME = "dangvantuan/sentence-camembert-base"  # french specific model (slow)
DEFAULT_SAVE_PATH = "final_faiss_index"
DEFAULT_DATA_PATH = "data"
# =========================

def load_imported_data(data_path):
    """Load previously imported documents"""
    print(f"Loading imported data from: {data_path}")
    
    documents_path = os.path.join(data_path, "documents.json")
    metadata_path = os.path.join(data_path, "import_metadata.json")
    
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"Documents file not found: {documents_path}")
    
    with open(documents_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    
    print(f"Loaded {len(docs)} documents")
    print(f"Import date: {metadata.get('import_date', 'Unknown')}")
    
    return docs, metadata

def get_embedding_text(doc):
    """Create text for embedding - MUST BE CONSISTENT!"""
    return "\n".join([
        doc.get("titre", ""),
        doc.get("description", ""),
        "Adresse : " + doc.get("adresse", ""),
        "Dates : " + ", ".join(doc.get("dates_affichage", [])),
    ])

def create_faiss_vectorstore_optimized(docs, model_name, save_path):
    """Optimized FAISS vectorstore creation with simple batch progress"""
    print("Creating FAISS vectorstore (optimized)...")
    
    start_time = time.time()
    
    try:
        # Step 1: Initialize embedding model
        print("   Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("   Embedding model loaded")
        
        # Step 2: Convert to LangChain documents
        print("   Converting documents...")
        langchain_docs = []
        
        for doc in tqdm(docs, desc="Preparing documents"):
            content = get_embedding_text(doc)
            langchain_docs.append(LangchainDocument(
                page_content=content,
                metadata=doc
            ))
        print("   Documents converted")
        
        # Step 3: Create FAISS index with batch progress
        print("   Creating FAISS index...")
        
        # Use a larger batch size for better performance but still show progress
        batch_size = 500
        total_batches = (len(langchain_docs) + batch_size - 1) // batch_size
        
        vectorstore = None
        for batch_num in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(langchain_docs))
            batch_docs = langchain_docs[start_idx:end_idx]
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch_docs, embedding_model)
            else:
                vectorstore.add_documents(batch_docs)
        
        print("   FAISS index created")
        
        # Save the vectorstore
        print("   Saving vectorstore...")
        vectorstore.save_local(save_path)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"FAISS vectorstore saved to: {save_path}")
        print(f"Total documents indexed: {len(docs)}")
        print(f"Time taken: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        return vectorstore
        
    except Exception as e:
        print(f"Error creating FAISS vectorstore: {e}")
        raise

def save_vectorization_metadata(docs, model_name, data_path, save_path):
    """Save metadata about the vectorization process"""
    metadata = {
        "model_name": model_name,
        "source_data_path": data_path,
        "document_count": len(docs),
        "vectorization_date": datetime.now().isoformat(),
        "embedding_fields": ["titre", "description", "adresse", "dates_affichage"]
    }
    
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "vectorization_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("Vectorization metadata saved")

def main(data_path=DEFAULT_DATA_PATH, save_path=DEFAULT_SAVE_PATH, model_name=MODEL_NAME):
    print("Starting FAISS vectorization...")
    print(f"Data source: {data_path}")
    print(f"Using model: {model_name}")
    print(f"Save path: {save_path}")
    
    try:
        # Step 1: Load imported data
        docs, import_metadata = load_imported_data(data_path)
        
        # Step 2: Create and save FAISS vectorstore
        vectorstore = create_faiss_vectorstore_optimized(docs, model_name, save_path)
        
        # Step 3: Save vectorization metadata
        save_vectorization_metadata(docs, model_name, data_path, save_path)
        
        # Step 4: Test the retriever
        print("Testing retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Quick test
        test_query = "Vin à Bordeaux"
        test_results = retriever.invoke(test_query)
        
        print(f"Test query '{test_query}' returned {len(test_results)} results")
        if test_results:
            print(f"   First result: {test_results[0].metadata['titre'][:50]}...")
        
        print("FAISS vectorization completed successfully!")
        
        return vectorstore
        
    except Exception as e:
        print(f"Error in vectorization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create FAISS vectorstore from imported data')
    parser.add_argument('--data-path', '-d', type=str, default=DEFAULT_DATA_PATH,
                       help='Path to imported data (default: data)')
    parser.add_argument('--save-path', '-s', type=str, default=DEFAULT_SAVE_PATH,
                       help='Path where to save the FAISS model (default: final_faiss_index)')
    parser.add_argument('--model', '-m', type=str, default=MODEL_NAME,
                       help=f'Embedding model to use (default: {MODEL_NAME})')
    args = parser.parse_args()
    
    vectorstore = main(data_path=args.data_path, save_path=args.save_path, model_name=args.model)