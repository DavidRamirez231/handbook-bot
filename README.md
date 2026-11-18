# Handbook Bot

A chatbot that answers questions from a long policy document, in English or Spanish, and shows the page each answer came from. It runs entirely on a laptop, so the document never leaves the computer.

I built this to learn how retrieval-augmented generation (RAG) works by making one from scratch instead of using a framework.

## What it does

- Reads a PDF and makes it searchable
- Answers questions using only that document
- Cites the page number for each fact
- Accepts a question in Spanish about an English document and replies in Spanish
- Says so when two parts of the document disagree
- Says it could not find an answer instead of making one up
- Includes a simple chat web page with a "looking it up" indicator

## How it works

1. **Read.** Pull the text out of the PDF, page by page.
2. **Chunk.** Cut the text into overlapping pieces of about 1,000 characters.
3. **Embed.** Turn each piece into a list of numbers that captures its meaning.
4. **Store.** Save the pieces and their numbers in a local `index/` folder.
5. **Search.** Turn the question into numbers the same way and find the five closest pieces.
6. **Answer.** Give those pieces to a local model with instructions to answer only from them.

No model is trained or fine-tuned. The model looks things up at question time.

## Tech

| Part | Tool |
|---|---|
| Local models | [Ollama](https://ollama.com) |
| Chat model | `qwen3:4b` or `llama3.2` |
| Embedding model | `embeddinggemma` |
| PDF reading | `pypdf` |
| Search | `numpy` (cosine similarity) |
| Web page | `Flask`, plain HTML, CSS, and JavaScript |

Built and tested on an M1 MacBook Air.

## Setup

Install [Ollama](https://ollama.com) and open it, then:

```
ollama pull qwen3:4b
ollama pull embeddinggemma

git clone https://github.com/YOUR-USERNAME/handbook-bot.git
cd handbook-bot
python3 -m venv .venv
source .venv/bin/activate
pip install ollama pypdf numpy flask
```

## Use

Put a PDF in the folder and build the index once:

```
python rag.py index "your-document.pdf"
```

Ask from the command line:

```
python rag.py ask "How many paid holidays are there?"
```

Or start the chat page and open http://127.0.0.1:8000:

```
python app.py
```

## What I learned

**Most fixes were in the instructions, not the code.** Three problems came up in testing, and none needed a change to the search:

| Problem | Fix |
|---|---|
| A Spanish question got "I could not find it" even though the right page was retrieved | Told the model the question may be in a different language than the document |
| The answer mixed English words into Spanish | Switched to a model with stronger multilingual ability |
| Two models gave two different numbers for the same question, because the document stated both | Told the model to show both and say they conflict |

**Search and answering fail in different ways.** When an answer is wrong, the first check is whether the right page was retrieved. If it was, the model or the prompt is the problem. If it was not, the search is.

**Small models trade speed for quality.** One model answered faster but in rougher Spanish. The other wrote clean Spanish but reasoned at length first, so the web page hides that reasoning and shows a status indicator.

**A human still has to check.** The bot can point out that two pages disagree. It cannot tell you which one is correct.

## Limits

- Each question is answered on its own. It does not remember earlier messages.
- Scanned PDFs with no selectable text need OCR first.
- Tables and forms inside PDFs may not be read well.
- Small local models make mistakes, so always check the cited page.

## Next steps

- Conversation memory for follow-up questions
- Support for more than one document at a time
- A test set of questions with known answers, to score changes
- A hosted version using a public housing policy document

## Privacy

The `.gitignore` keeps PDFs and the `index/` folder out of this repository, because the index contains the text of whatever document you load. Only use documents you have the right to use.

## About

Built by David Ramirez, a housing case manager learning to build AI tools for affordable housing and nonprofit work. I used an AI assistant as a pair programmer while building this.
