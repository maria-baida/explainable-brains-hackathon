"""
chat_server.py
==============

Tiny FastAPI server that backs the chat panel in volcano_mockup.html.

Why this exists:
    volcano_mockup.html is a static page (Plotly inline). The chat panel
    inside it needs a live backend to call Claude. This server:

      1. Serves the HTML at  GET  /
      2. Handles chat at      POST /chat   {"message": "..."} → {"response": "..."}

Auth:
    Uses claude-agent-sdk with Claude Code's OAuth login (your Max plan).
    No ANTHROPIC_API_KEY required.

Run:
    python chat_server.py
    # then open http://localhost:8000
"""

import asyncio
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
)
from claude_agent_sdk import query as claude_query


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Explainable Brains chat backend")

# Permissive CORS in case the HTML is opened via file:// or a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parent
HTML_PATH = REPO_ROOT / "volcano_mockup.html"
STATS_PATH = REPO_ROOT / "data_b" / "statistics.csv"


# ── Build the chat system prompt once at startup ────────────────────────────
def _build_system_prompt() -> str:
    """Bake the top 80 most-significant regions into the system prompt so the
    chat can answer without making tool calls."""
    if not STATS_PATH.exists():
        return "Statistics CSV missing — chat answers will be generic."

    stats = pd.read_csv(STATS_PATH)
    top = (
        stats.dropna(subset=["log2_fold_change", "p_value"])
        .sort_values("p_value")
        .head(80)
    )

    rows = []
    for _, r in top.iterrows():
        acronym = str(r.get("acronym", ""))[:10]
        name = str(r.get("region_name", ""))[:42]
        rows.append(
            f"  {acronym:10s} | {name:42s} | "
            f"log2FC={r['log2_fold_change']:+.2f} | p={r['p_value']:.4f}"
        )
    data_block = "\n".join(rows)

    return f"""You are a neuroscience research assistant for an Ozempic / Semaglutide brain-imaging study.

Experimental setup:
- G001 (Vehicle): control mice given saline
- G002 (Semaglutide): mice given Semaglutide (active ingredient in Ozempic / Wegovy)
- Marker: c-Fos, expressed by recently active neurons — proxy for neural activation
- 1,356 brain regions analyzed total

Top 80 regions by smallest uncorrected p-value:

```
  acronym    | region_name                                | log2FC      | p
{data_block}
```

How to read this:
- Positive log2FC = MORE active in Semaglutide
- Negative log2FC = LESS active in Semaglutide
- With 1356 tests, multiple-testing correction wipes out everything, so we
  use uncorrected p<0.05 as the working threshold.

When answering:
1. Use your neuroanatomy knowledge to identify which regions ABOVE are
   relevant. Don't invent regions that aren't in the table.
2. Cite acronyms with numbers — e.g., "NTS (nucleus of the solitary tract),
   log2FC = +2.77, p = 0.006".
3. Connect findings to Semaglutide's mechanism (gut→brainstem satiety,
   amygdalar food valuation, reward dampening).
4. Be concise — 3 to 5 sentences. Plain prose, no markdown headings."""


SYSTEM_PROMPT = _build_system_prompt()


# ── Chat endpoint ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


async def _run_chat(prompt: str) -> str:
    """Send one prompt through Claude Code OAuth, collect all text blocks."""
    opts = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=[],  # pure chat — no Bash/FileRead/WebFetch
    )
    collected: list[str] = []
    seen_types: list[str] = []
    async for msg in claude_query(prompt=prompt, options=opts):
        seen_types.append(type(msg).__name__)
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    collected.append(block.text)
    text = "".join(collected).strip()
    if not text:
        return f"(No text response. SDK returned: {seen_types})"
    return text


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        response_text = await _run_chat(req.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        # Surface the error to the client so we can see it in the browser console
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ── Serve the HTML at root ───────────────────────────────────────────────────
@app.get("/")
async def root():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail=f"{HTML_PATH.name} not found")
    return FileResponse(HTML_PATH, media_type="text/html")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting chat server on http://localhost:8000")
    print("Open that URL in your browser — it will serve volcano_mockup.html")
    uvicorn.run(app, host="0.0.0.0", port=8000)
