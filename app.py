"""
A chat web page for your handbook bot.

It uses the index you already built with rag.py, and it runs only on
your own computer.

Setup (one time):
    pip install flask

Run:
    python app.py

Then open http://127.0.0.1:8000 in your browser.
Press Control + C in Terminal to stop it.
"""

import json
import sys

import numpy as np
import ollama
from flask import Flask, Response, request

import rag  # reuses the settings and the embed() function from rag.py

if not (rag.INDEX_DIR / "vectors.npy").exists():
    sys.exit("No index yet. Run: python rag.py index yourfile.pdf")

chunks = json.loads((rag.INDEX_DIR / "chunks.json").read_text())
vectors = np.load(rag.INDEX_DIR / "vectors.npy")

app = Flask(__name__)


def build_prompt(question, best):
    context = "\n\n".join(
        f"[{chunks[i]['file']}, page {chunks[i]['page']}]\n{chunks[i]['text']}"
        for i in best
    )
    prompt = (
        "Answer the question using only the excerpts below. "
        "The question may be in a different language than the excerpts. "
        "Translate as needed and reply in the language of the question. "
        "If the excerpts disagree with each other, show both and say they conflict. "
        "After each fact, give the page number in brackets, like [page 44]. "
        "Keep the answer short: two to four sentences. "
        "Only say you could not find the answer if none of the excerpts are relevant.\n\n"
        f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
    )
    if "qwen" in rag.CHAT_MODEL.lower():
        prompt += " /no_think"  # asks Qwen to skip its long thinking step
    return prompt


def ask_model(prompt):
    """Get the whole reply from the model as one piece of text."""
    messages = [{"role": "user", "content": prompt}]
    try:
        reply = ollama.chat(model=rag.CHAT_MODEL, messages=messages, think=False)
    except (ollama.ResponseError, TypeError):
        reply = ollama.chat(model=rag.CHAT_MODEL, messages=messages)
    return reply["message"]["content"] or ""


def remove_thinking(text):
    """Some models write their private reasoning first. Keep only the answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        text = ""  # reasoning never finished, so there is no answer to show
    return text.replace("<think>", "").replace("**", "").strip()


def event(**data):
    """One line of news for the web page."""
    return json.dumps(data) + "\n"


def answer(question):
    """Tell the page what is happening at each step, then send the answer."""
    yield event(status="Searching the handbook")
    query = rag.embed([question])[0]
    best = np.argsort(vectors @ query)[::-1][: rag.TOP_K]
    pages = sorted({chunks[i]["page"] for i in best})

    page_list = ", ".join(str(p) for p in pages)
    yield event(status=f"Reading pages {page_list}")
    prompt = build_prompt(question, best)

    yield event(status="Writing the answer")
    try:
        text = remove_thinking(ask_model(prompt))
    except Exception as error:  # show the problem on the page instead of hanging
        yield event(error=f"The model could not answer: {error}")
        return

    if not text:
        text = "I could not put together an answer. Please try asking again."
    yield event(answer=text, pages=pages)


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return Response(event(error="Please type a question."),
                        mimetype="application/x-ndjson")
    return Response(answer(question), mimetype="application/x-ndjson")


@app.get("/")
def home():
    return PAGE


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Handbook Assistant</title>
<style>
  :root { --bg:#f6f5f2; --card:#ffffff; --ink:#1d1d1f; --soft:#6b6b70;
          --line:#e3e1dc; --me:#1f4e79; --me-ink:#ffffff; --chip:#eef2f6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#17181a; --card:#222326; --ink:#ececee; --soft:#9a9aa0;
            --line:#34353a; --me:#3b7cc0; --me-ink:#ffffff; --chip:#2c2e33; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .wrap { max-width: 760px; margin: 0 auto; height: 100vh;
          display: flex; flex-direction: column; padding: 0 16px; }
  header { padding: 20px 4px 12px; border-bottom: 1px solid var(--line); }
  header h1 { margin:0; font-size: 20px; }
  header p { margin: 4px 0 0; color: var(--soft); font-size: 14px; }
  #chat { flex: 1; overflow-y: auto; padding: 20px 4px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 14px;
         margin: 0 0 12px; white-space: pre-wrap; word-wrap: break-word; }
  .me  { background: var(--me); color: var(--me-ink); margin-left: auto;
         border-bottom-right-radius: 4px; }
  .bot { background: var(--card); border: 1px solid var(--line);
         border-bottom-left-radius: 4px; width: fit-content; }
  .hint { color: var(--soft); text-align: center; margin-top: 40px; }

  .working { display: flex; align-items: center; gap: 10px; color: var(--soft); }
  .dots { display: flex; gap: 5px; }
  .dots span { width: 8px; height: 8px; border-radius: 50%; background: var(--soft);
               animation: bounce 1.2s infinite ease-in-out; }
  .dots span:nth-child(2) { animation-delay: .15s; }
  .dots span:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { 0%, 80%, 100% { opacity: .3; transform: translateY(0); }
                      40% { opacity: 1; transform: translateY(-4px); } }
  @media (prefers-reduced-motion: reduce) { .dots span { animation: none; } }

  .pages { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;
           padding-top: 10px; border-top: 1px solid var(--line);
           font-size: 13px; color: var(--soft); align-items: center; }
  .pages b { font-weight: 600; }
  .chip { background: var(--chip); border-radius: 999px; padding: 2px 10px; color: var(--ink); }

  form { display: flex; gap: 8px; padding: 12px 0 20px;
         border-top: 1px solid var(--line); }
  input { flex: 1; padding: 12px 14px; border-radius: 10px; font: inherit;
          border: 1px solid var(--line); background: var(--card); color: var(--ink); }
  button { padding: 0 20px; border: 0; border-radius: 10px; font: inherit;
           background: var(--me); color: var(--me-ink); cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Handbook Assistant</h1>
    <p>Answers come only from the document, with page numbers. Ask in English or Spanish.</p>
  </header>
  <div id="chat"><p class="hint" id="hint">Ask a question to get started.</p></div>
  <form id="form">
    <input id="q" placeholder="Type your question" autocomplete="off" autofocus>
    <button id="send">Send</button>
  </form>
</div>
<script>
  const chat = document.getElementById("chat");
  const form = document.getElementById("form");
  const input = document.getElementById("q");
  const send = document.getElementById("send");

  function scrollDown() { chat.scrollTop = chat.scrollHeight; }

  function bubble(kind, text) {
    const el = document.createElement("div");
    el.className = "msg " + kind;
    el.textContent = text;
    chat.appendChild(el);
    scrollDown();
    return el;
  }

  // The three bouncing dots with a short status line next to them.
  function showWorking(el, status) {
    el.textContent = "";
    const row = document.createElement("div");
    row.className = "working";
    const dots = document.createElement("div");
    dots.className = "dots";
    for (let i = 0; i < 3; i++) dots.appendChild(document.createElement("span"));
    const label = document.createElement("span");
    label.textContent = status;
    row.appendChild(dots);
    row.appendChild(label);
    el.appendChild(row);
    scrollDown();
  }

  function showAnswer(el, text, pages) {
    el.textContent = text;
    if (pages && pages.length) {
      const row = document.createElement("div");
      row.className = "pages";
      const label = document.createElement("b");
      label.textContent = "Pages:";
      row.appendChild(label);
      for (const page of pages) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = page;
        row.appendChild(chip);
      }
      el.appendChild(row);
    }
    scrollDown();
  }

  function handle(line, reply) {
    if (!line.trim()) return;
    const data = JSON.parse(line);
    if (data.status) showWorking(reply, data.status);
    if (data.error) reply.textContent = data.error;
    if (data.answer) showAnswer(reply, data.answer, data.pages);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    const hint = document.getElementById("hint");
    if (hint) hint.remove();

    bubble("me", question);
    const reply = bubble("bot", "");
    showWorking(reply, "Searching the handbook");
    input.value = "";
    send.disabled = true;

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let waiting = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        waiting += decoder.decode(value, { stream: true });
        const lines = waiting.split("\\n");
        waiting = lines.pop();
        for (const line of lines) handle(line, reply);
      }
      handle(waiting, reply);
    } catch (error) {
      reply.textContent = "Something went wrong. Is Ollama running? Check the Terminal window.";
    }
    send.disabled = false;
    input.focus();
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser. Press Control + C to stop.")
    app.run(host="127.0.0.1", port=8000, debug=False)
