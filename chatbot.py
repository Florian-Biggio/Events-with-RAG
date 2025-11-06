import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
import argparse

def list_available_models():
    """List all available FAISS models"""
    models = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item != ".git":  # Skip .git folder
            # Check for FAISS-specific file patterns
            faiss_files = [
                'index.faiss', 
                'faiss.index',
                'index_faiss.bin'
            ]
            # Check if it has actual FAISS index files, not just metadata
            has_faiss_index = any(os.path.exists(os.path.join(item, f)) for f in faiss_files)
            
            if has_faiss_index:
                models.append(item)
    
    return models

def load_secrets(file_path='secrets.json'):
    """Load API keys from secrets file"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Secrets file '{file_path}' not found.")
        print("Please create a secrets.json file with your CHAT_MISTRAL_AI_KEY")
        return None

def detect_model_type(faiss_path):
    """Detect what type of model this is (LangChain FAISS or custom)"""
    if os.path.exists(os.path.join(faiss_path, "index.faiss")):
        return "langchain_faiss"
    elif os.path.exists(os.path.join(faiss_path, "index_faiss.bin")):
        return "custom_faiss"
    else:
        return "unknown"

def load_langchain_faiss(faiss_path):
    """Load standard LangChain FAISS format"""
    # Look for metadata to get model name
    metadata_path = os.path.join(faiss_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model_name = metadata.get("model_name")
    else:
        # Try to infer model name from other files
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Default
    
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    vectorstore = FAISS.load_local(
        faiss_path, 
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore, metadata

def load_custom_faiss(faiss_path):
    """Load your custom FAISS format (from your original script)"""
    import faiss
    import numpy as np
    
    # Load the index
    index = faiss.read_index(os.path.join(faiss_path, "index_faiss.bin"))
    
    # Load documents
    with open(os.path.join(faiss_path, "documents.json"), "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    # Load metadata
    with open(os.path.join(faiss_path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Convert to LangChain format
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document as LangchainDocument
    
    embedding_model = HuggingFaceEmbeddings(model_name=meta.get("model"))
    
    # Convert documents
    def get_embedding_text(doc):
        return "\n".join([
            doc.get("titre", ""),
            doc.get("description", ""),
            "Adresse : " + doc.get("adresse", ""),
            "Dates : " + ", ".join(doc.get("dates affichage", [])),
        ])
    
    langchain_docs = []
    for doc in docs:
        content = get_embedding_text(doc)
        langchain_docs.append(LangchainDocument(
            page_content=content,
            metadata=doc
        ))
    
    # Create FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=FAISS._build_docstore(langchain_docs),
        index_to_docstore_id={i: i for i in range(len(langchain_docs))}
    )
    
    metadata = {
        "model_name": meta.get("model"),
        "document_count": len(docs),
        "source": "custom_faiss"
    }
    
    return vectorstore, metadata

def load_rag_bot(faiss_path="final_faiss_index"):
    """Load the RAG bot - supports multiple formats"""
    print("📥 Loading RAG bot...")
    
    try:
        model_type = detect_model_type(faiss_path)
        print(f"   Detected model type: {model_type}")
        
        if model_type == "langchain_faiss":
            vectorstore, metadata = load_langchain_faiss(faiss_path)
        elif model_type == "custom_faiss":
            vectorstore, metadata = load_custom_faiss(faiss_path)
        else:
            print(f"Unknown model type in '{faiss_path}'")
            return None, None
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        print("RAG bot loaded successfully!")
        print(f"   Documents: {metadata.get('document_count', 'Unknown')}")
        print(f"   Model: {metadata.get('model_name', 'Unknown')}")
        print(f"   Source: {metadata.get('source', 'Unknown')}")
        
        return retriever, metadata
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print(f"   Please check that '{faiss_path}' contains a valid FAISS index")
        return None, None
    except Exception as e:
        print(f"Error loading RAG bot: {e}")
        return None, None

def create_qa_chain(retriever, mistral_api_key):
    """Create the QA chain with a custom prompt"""
    
    prompt_template = """Tu es un assistant spécialisé dans les événements en Nouvelle-Aquitaine. 
Utilise les informations suivantes pour répondre à la question. Si tu ne sais pas, dis que tu ne sais pas.

Informations disponibles:
{context}

Question: {question}

Réponse utile:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatMistralAI(
        api_key=mistral_api_key, 
        model="mistral-medium",
        temperature=0.1
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def format_source_documents(source_docs, max_chars=200):
    """Format source documents for display"""
    formatted_sources = []
    for i, doc in enumerate(source_docs, 1):
        source_info = f"\nSource {i}: {doc.metadata.get('titre', 'Unknown')}"
        if doc.metadata.get('adresse'):
            source_info += f"\n   {doc.metadata['adresse']}"
        if doc.metadata.get('dates_affichage'):
            dates = doc.metadata['dates_affichage'][:2]
            source_info += f"\n    {', '.join(dates)}"
        elif doc.metadata.get('dates affichage'):  # Your original format
            dates = doc.metadata['dates affichage'][:2]
            source_info += f"\n    {', '.join(dates)}"
        formatted_sources.append(source_info)
    return "\n".join(formatted_sources)

def interactive_chat(qa_chain):
    """Interactive chat interface"""
    print("\n" + "="*60)
    print("Chatbot des Événements - Nouvelle-Aquitaine")
    print("="*60)
    print("Posez-moi des questions sur les événements dans la région!")
    print("Tapez 'quit' ou 'exit' pour quitter.")
    print("="*60)
    
    while True:
        try:
            query = input("\nVous: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Au revoir!")
                break
                
            if not query:
                continue
                
            print("Recherche en cours...")
            result = qa_chain.invoke(query)
            
            print(f"\nAssistant: {result['result']}")
            
            if result.get('source_documents'):
                print(f"\nSources utilisées:{format_source_documents(result['source_documents'])}")
                
        except KeyboardInterrupt:
            print("\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Chatbot des événements en Nouvelle-Aquitaine')
    parser.add_argument('--query', '-q', type=str, help='Run a single query instead of interactive mode')
    parser.add_argument('--faiss-path', '-p', type=str, default='final_faiss_index', 
                       help='Path to FAISS index directory')
    parser.add_argument('--list-models', '-l', action='store_true', 
                       help='List available models and exit')
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"   {model}")
        else:
            print("No FAISS models found in current directory")
        return
    
    # Load secrets
    secrets = load_secrets()
    if not secrets:
        return
    
    mistral_api_key = secrets.get('CHAT_MISTRAL_AI_KEY')
    if not mistral_api_key:
        print("CHAT_MISTRAL_AI_KEY not found in secrets.json")
        return
    
    # Load RAG bot
    retriever, metadata = load_rag_bot(args.faiss_path)
    if not retriever:
        return
    
    # Create QA chain
    qa_chain = create_qa_chain(retriever, mistral_api_key)
    
    # Run in appropriate mode
    if args.query:
        print(f"Question: {args.query}")
        print("Recherche en cours...")
        result = qa_chain.invoke(args.query)
        print(f"\nAssistant: {result['result']}")
        if result.get('source_documents'):
            print(f"\nSources utilisées:{format_source_documents(result['source_documents'])}")
    else:
        interactive_chat(qa_chain)

if __name__ == "__main__":
    main()
