import os
import textwrap
from typing import List

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
try:
    from langchain.schema import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

SPEECH_FILE = "speech.txt"
PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "ambedkar_speech"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3 

def simple_text_loader(path: str, encoding: str = "utf-8") -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding=encoding) as f:
        return [Document(page_content=f.read(), metadata={"source": path})]


def create_or_load_vectorstore() -> Chroma:
    """Create or load Chroma vectorstore."""
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"[info] Loading existing Chroma DB from '{PERSIST_DIR}'...")
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vdb = Chroma(persist_directory=PERSIST_DIR,
                     collection_name=CHROMA_COLLECTION_NAME,
                     embedding_function=emb)
        return vdb

    print("[info] Building Chroma DB from speech.txt...")
    docs = simple_text_loader(SPEECH_FILE)

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator=True
    )

    try:
        docs_chunks = splitter.split_documents(docs)
    except Exception as e:
        print("[warn] splitter.split_documents failed, performing naive split:", e)
        text = docs[0].page_content
        chunks = []
        i = 0
        while i < len(text):
            chunk_text = text[i:i + CHUNK_SIZE]
            chunks.append(Document(page_content=chunk_text))
            i += CHUNK_SIZE - CHUNK_OVERLAP
        docs_chunks = chunks

    print(f"[info] Created {len(docs_chunks)} chunks. Computing embeddings and storing in Chroma...")
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vdb = Chroma.from_documents(
        documents=docs_chunks,
        embedding=emb,
        persist_directory=PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    vdb.persist()
    print("[info] Vectorstore built and persisted.")
    return vdb


def retrieve_top_k(vdb: Chroma, query: str, k: int = TOP_K) -> List[Document]:
    """
    Use Chroma's similarity search via the retriever interface to get top-k docs.
    We try multiple methods safely depending on API availability.
    """
    try:
        retriever = vdb.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return docs
    except Exception:
        pass
    try:
        docs = vdb.similarity_search(query, k=k)
        return docs
    except Exception:
        pass

    return []


def build_prompt(context_chunks: List[Document], question: str) -> str:
    """
    Build a concise prompt that provides context and asks the model to answer
    based only on that context.
    """
    if not context_chunks:
        context_text = "(no context found)"
    else:
        pieces = []
        for i, d in enumerate(context_chunks, start=1):
            snippet = d.page_content.strip().replace("\n", " ")
            pieces.append(f"Context {i}:\n{snippet}")
        context_text = "\n\n".join(pieces)

    prompt = textwrap.dedent(f"""
    You are a helpful assistant. Use ONLY the provided context to answer the question.
    If the answer is not contained in the context, say "I don't know based on the given text."

    CONTEXT:
    {context_text}

    QUESTION:
    {question}

    Answer concisely and base your answer only on the context above.
    """).strip()
    return prompt


def answer_with_ollama(llm: Ollama, prompt: str) -> str:
    """
    Call the Ollama wrapper. Attempt common call styles and return text.
    """
    try:
        out = llm(prompt)
        if isinstance(out, dict):
           
            for k in ("text", "result", "output"):
                if k in out:
                    return out[k]
            return str(out)
        return str(out)
    except Exception:
        pass

    try:
        gen = llm.generate([prompt])
        
        try:
            
            generations = getattr(gen, "generations", None)
            if generations:
                return generations[0][0].text
        except Exception:
            pass
        return str(gen)
    except Exception:
        pass

    try:
        pred = llm.predict(prompt)
        return str(pred)
    except Exception:
        pass

    return "Error: failed to call Ollama LLM with available API."


def cli_loop(vdb: Chroma):
    """Read-eval-print loop: get question, retrieve context, build prompt, call LLM, print answer."""
    print("\nAmbedkarGPT — Ask questions about the speech. Type 'exit' or 'quit' to stop.\n")
    llm = Ollama(model="mistral")  
    while True:
        try:
            q = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        docs = retrieve_top_k(vdb, q, k=TOP_K)
        prompt = build_prompt(docs, q)
        ans = answer_with_ollama(llm, prompt)
        print("\nA:", ans)
        if docs:
            print("\n[context snippets used]:")
            for i, d in enumerate(docs, start=1):
                s = d.page_content.strip().replace("\n", " ")
                print(f"  {i}. {s[:400]}{'...' if len(s)>400 else ''}")
        print("-" * 60)


def main():
    vdb = create_or_load_vectorstore()
    cli_loop(vdb)


if __name__ == "__main__":
    main()
