"""
Local handbook chatbot (RAG) for a Mac with Ollama.

Everything stays on your computer. Nothing is uploaded.

Setup (one time):
    ollama pull llama3.2
    ollama pull embeddinggemma
    pip install ollama pypdf numpy

Use:
    python rag.py index handbook.pdf second_doc.pdf
    python rag.py ask "How many sick days do new employees get?"
"""

import json
import sys
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

CHAT_MODEL = "qwen3:4b"          # small model that fits an 8 GB Mac
EMBED_MODEL = "embeddinggemma"   # turns text into numbers
INDEX_DIR = Path("index")
CHUNK_SIZE = 1000                # characters per chunk
CHUNK_OVERLAP = 150              # characters shared between neighbours
TOP_K = 5                        # how many chunks to show the model
BATCH = 32


def read_chunks(pdf_path):
    """Step 1 and 2: pull the text out of a PDF and cut it into chunks."""
    reader = PdfReader(pdf_path)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE]
            if len(piece) > 50:
                chunks.append({
                    "file": Path(pdf_path).name,
                    "page": page_number,
                    "text": piece,
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    """Step 3: turn text into vectors, a batch at a time."""
    vectors = []
    for i in range(0, len(texts), BATCH):
        response = ollama.embed(model=EMBED_MODEL, input=texts[i:i + BATCH])
        vectors.extend(response["embeddings"])
        print(f"  embedded {min(i + BATCH, len(texts))} of {len(texts)}", end="\r")
    print()
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def build_index(pdf_paths):
    """Step 4: save the chunks and their vectors to disk."""
    chunks = []
    for path in pdf_paths:
        found = read_chunks(path)
        print(f"{path}: {len(found)} chunks")
        chunks.extend(found)
    if not chunks:
        sys.exit("No text found. If the PDF is a scan, it needs OCR first.")
    vectors = embed([c["text"] for c in chunks])
    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    np.save(INDEX_DIR / "vectors.npy", vectors)
    print(f"Saved {len(chunks)} chunks to {INDEX_DIR}/")


def ask(question):
    """Step 5 and 6: find the closest chunks, then answer from them."""
    if not (INDEX_DIR / "vectors.npy").exists():
        sys.exit("No index yet. Run: python rag.py index yourfile.pdf")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
    vectors = np.load(INDEX_DIR / "vectors.npy")

    query = embed([question])[0]
    scores = vectors @ query
    best = np.argsort(scores)[::-1][:TOP_K]

    context = "\n\n".join(
        f"[{chunks[i]['file']}, page {chunks[i]['page']}]\n{chunks[i]['text']}"
        for i in best
    )
    prompt = (
        "Answer the question using only the excerpts below. "
        "The question may be in a different language than the excerpts. "
        "Translate as needed and reply in the language of the question. "
       "If the excerpts disagree with each other, show both and say they conflict. "
        "After each fact, give the file and page in brackets. "
        "Only say you could not find the answer if none of the excerpts are relevant.\n\n"
        f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
    )

    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print("\n\nSources checked:")
    for i in best:
        print(f"  {chunks[i]['file']}, page {chunks[i]['page']}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "index":
        build_index(sys.argv[2:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
