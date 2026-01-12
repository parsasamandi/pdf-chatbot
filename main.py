from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import io
import requests

app = FastAPI()

# Initialize
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("pdf_docs")

# Store current PDF metadata
current_pdf = {"name": "", "total_pages": 0}

def extract_pdf(file_bytes):
    """Extract text from PDF with page numbers"""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            pages_text.append({
                "page": page_num,
                "text": text
            })
    
    return pages_text

def chunk_text(pages_text, chunk_size=500, overlap=50):
    """Split text into chunks while preserving page numbers"""
    chunks = []
    
    for page_data in pages_text:
        text = page_data["text"]
        page_num = page_data["page"]
        
        # Simple chunking by characters
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num
                })
            
            start = end - overlap
    
    return chunks

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF"""
    # Read file
    contents = await file.read()
    
    # Extract text with page numbers
    pages_text = extract_pdf(contents)
    
    # Store metadata
    current_pdf["name"] = file.filename
    current_pdf["total_pages"] = len(pages_text)
    
    # Chunk text
    chunks = chunk_text(pages_text)
    
    # Clear existing collection and recreate
    global collection
    try:
        chroma_client.delete_collection("pdf_docs")
    except:
        pass
    collection = chroma_client.create_collection("pdf_docs")
    
    # Generate embeddings and store
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk["text"]).tolist()
        
        collection.add(
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"]}],
            ids=[f"chunk_{i}"]
        )
    
    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "total_pages": len(pages_text),
        "total_chunks": len(chunks)
    }

def retrieve_chunks(question, top_k=3):
    """Retrieve most relevant chunks"""
    query_embedding = embedding_model.encode(question).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "page": results['metadatas'][0][i]['page']
        })
    
    return chunks

def ask_ollama(prompt):
    """Call Ollama API"""
    OLLAMA_API = "http://localhost:11434/api/generate"
    
    response = requests.post(
        OLLAMA_API,
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )
    
    return response.json()["response"]

@app.post("/chat")
async def chat(question: str):
    """Answer questions using RAG"""
    # Retrieve relevant chunks
    chunks = retrieve_chunks(question, top_k=3)
    
    # Build context
    context = "\n\n".join([
        f"[Page {chunk['page']}]: {chunk['text']}" 
        for chunk in chunks
    ])
    
    # Build prompt
    prompt = f"""You are a helpful assistant answering questions about a PDF document.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context provided
- If the answer is not in the context, say "I don't have enough information to answer that"
- Be concise and clear
- Mention which page(s) you found the information on

Answer:"""
    
    # Get answer from Ollama
    answer = ask_ollama(prompt)
    
    # Return answer with sources
    return {
        "answer": answer,
        "sources": [{"page": chunk["page"], "text": chunk["text"][:200] + "..."} 
                   for chunk in chunks]
    }

@app.get("/")
async def home():
    try:
        return HTMLResponse(open("index.html").read())
    except:
        return {"message": "Upload a PDF to /upload endpoint"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)