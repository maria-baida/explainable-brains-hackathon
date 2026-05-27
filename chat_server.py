"""
chat_server.py
==============

FastAPI server backing the chat panel in volcano_mockup.html.

  GET  /        → serves volcano_mockup.html
  POST /chat    → {"message": "..."} → {"response": "..."}

Setup:
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python chat_server.py
    # then open http://localhost:8000
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import anthropic
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Explainable Brains chat backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve NIfTI / atlas files for the 3D brain viewer (matches Vercel's static path)
if DATA_PATH.exists():
    app.mount("/data_b", StaticFiles(directory=str(DATA_PATH)), name="data_b")

REPO_ROOT = Path(__file__).resolve().parent
HTML_PATH = REPO_ROOT / "volcano_mockup.html"
STATS_PATH = REPO_ROOT / "data_b" / "statistics.csv"
DATA_PATH = REPO_ROOT / "data_b"


# ── Build system prompt once at startup ──────────────────────────────────────
def _build_system_prompt() -> str:
    if not STATS_PATH.exists():
        return "Statistics CSV missing — chat answers will be generic."

    stats = pd.read_csv(STATS_PATH)
    # Use all regions with valid stats, sorted by p_value (uncorrected)
    top = (
        stats.dropna(subset=["log2_fold_change", "p_value"])
        .sort_values("p_value")
        .head(100)
    )

    rows = []
    for _, r in top.iterrows():
        acronym = str(r.get("acronym", ""))[:10]
        name = str(r.get("region_name", ""))[:42]
        sig = "*" if r.get("significant_uncorrected", False) else " "
        rows.append(
            f"{sig} {acronym:10s} | {name:42s} | "
            f"log2FC={r['log2_fold_change']:+.2f} | p={r['p_value']:.4f}"
        )
    data_block = "\n".join(rows)

    return f"""You are a neuroscience research assistant for an Ozempic / Semaglutide brain-imaging study.

Experimental setup:
- G001 (Vehicle): control mice given saline
- G002 (Semaglutide): mice given Semaglutide (active ingredient in Ozempic / Wegovy)
- Marker: c-Fos, a protein expressed by recently active neurons — proxy for neural activation
- 833 brain regions analyzed

Top 100 regions by uncorrected p-value (* = significant at p<0.05):

```
  acronym    | region_name                                | log2FC      | p
{data_block}
```

How to read:
- Positive log2FC = MORE active in Semaglutide mice
- Negative log2FC = LESS active in Semaglutide mice
- Multiple-testing correction wipes out all significance, so we use uncorrected p<0.05 (37 regions pass)

When answering:
1. Only cite regions from the table above — don't invent data.
2. Give acronym + full name + numbers: "NTS (nucleus of the solitary tract), log2FC=+2.77, p=0.006".
3. Connect findings to Semaglutide's known mechanism (gut→brainstem satiety signalling, reward dampening, amygdala food valuation).
4. Be concise — 3–5 sentences, plain prose, no markdown headings."""


SYSTEM_PROMPT = _build_system_prompt()


# ── Chat endpoint ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def _run_chat(prompt: str) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-...",
        )
    try:
        response_text = _run_chat(req.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ── Serve HTML at root ────────────────────────────────────────────────────────
@app.get("/")
async def root():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail=f"{HTML_PATH.name} not found")
    return FileResponse(HTML_PATH, media_type="text/html")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)

    print("Starting chat server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
