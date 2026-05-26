"""
Challenge B — Natural language agent
Uses Claude (Anthropic API) to answer questions about the brain imaging data.

Setup:
    export ANTHROPIC_API_KEY=sk-ant-...   (your key from console.anthropic.com)

Run:
    python challenge_b_agent.py

Example questions:
    "Which brain regions are most upregulated by semaglutide?"
    "Which regions are involved in hunger regulation, and how do they compare?"
    "What are the top 5 most statistically significant regions?"
    "Explain what the NTS region does and whether semaglutide affects it"
"""

import os
import json
import pandas as pd
import numpy as np
import anthropic
from bucket_access.bucket_utils import download_file

# ── Model — change to claude-haiku-4-5 to save credits on simple queries ─────
MODEL = "claude-opus-4-7"

# ── Load data (download once, cache locally) ──────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs("data_b", exist_ok=True)

    quant_path = "data_b/quantification.csv"
    stats_path = "data_b/statistics.csv"

    if not os.path.exists(quant_path):
        print("Downloading quantification data...")
        download_file(
            "challengeB/tabular_data_quantification/cfos_object_density_quantification.csv",
            quant_path
        )
    if not os.path.exists(stats_path):
        print("Downloading statistics data...")
        download_file(
            "challengeB/tabular_data_quantification/cfos_object_density_statistics_G002_vs_G001.csv",
            stats_path
        )

    quant = pd.read_csv(quant_path)
    stats = pd.read_csv(stats_path)
    print(f"Loaded: {stats.shape[0]} brain regions, {quant.shape[0]} animals")
    return quant, stats


# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_top_regions",
        "description": (
            "Get the top N brain regions most differentially activated between "
            "semaglutide (G002) and vehicle (G001) animals. "
            "Returns region name, acronym, log2 fold change, and corrected p-value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "both"],
                    "description": "'up' = upregulated in semaglutide, 'down' = downregulated, 'both' = sorted by absolute fold change"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of regions to return (default 10)"
                },
                "significant_only": {
                    "type": "boolean",
                    "description": "If true, only return statistically significant regions (uncorrected p<0.05). With 1356 regions tested, no region clears multiple-testing correction, so we use uncorrected p-values."
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "search_regions",
        "description": (
            "Search for specific brain regions by name or acronym. "
            "Returns their statistics (fold change, p-values, mean densities). "
            "Useful when the user mentions a specific brain area by name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Region name or acronym to search for (partial match supported)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_summary_stats",
        "description": "Get a high-level summary of the dataset: number of regions, how many are significant, overall distribution of fold changes.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


# ── Tool execution ────────────────────────────────────────────────────────────
def execute_tool(name: str, inputs: dict, stats_df: pd.DataFrame, quant_df: pd.DataFrame) -> str:
    if name == "get_top_regions":
        direction = inputs.get("direction", "both")
        n = inputs.get("n", 10)
        sig_only = inputs.get("significant_only", False)

        df = stats_df.copy()
        # Coerce the significance boolean (CSV loads it as float64 with NaNs)
        df["significant_uncorrected"] = df["significant_uncorrected"].fillna(False).astype(bool)
        if sig_only:
            df = df[df["significant_uncorrected"]]

        if direction == "up":
            df = df.nlargest(n, "log2_fold_change")
        elif direction == "down":
            df = df.nsmallest(n, "log2_fold_change")
        else:
            df = df.reindex(df["log2_fold_change"].abs().nlargest(n).index)

        cols = ["acronym", "region_name", "log2_fold_change", "p_value",
                "significant_uncorrected", "mean_A", "mean_B", "hierarchy_level"]
        result = df[cols].round(4).to_dict(orient="records")
        return json.dumps(result, indent=2)

    elif name == "search_regions":
        query = inputs["query"].lower()
        mask = (
            stats_df["region_name"].str.lower().str.contains(query, na=False) |
            stats_df["acronym"].str.lower().str.contains(query, na=False)
        )
        matches = stats_df[mask]
        if matches.empty:
            return json.dumps({"message": f"No regions found matching '{query}'"})
        cols = ["acronym", "region_name", "log2_fold_change", "fold_change",
                "p_corrected", "significant_corrected", "mean_A", "mean_B",
                "median_A", "median_B"]
        return matches[cols].round(4).to_dict(orient="records").__str__()

    elif name == "get_summary_stats":
        # Coerce significance columns to bool (CSV loads them as float64 with NaNs)
        sig_corr = stats_df["significant_corrected"].fillna(False).astype(bool)
        sig_uncorr = stats_df["significant_uncorrected"].fillna(False).astype(bool)
        sig_corrected = int(sig_corr.sum())
        sig_uncorrected = int(sig_uncorr.sum())
        n_animals = len(quant_df)
        n_g001 = int((quant_df["group_nr"] == "G001").sum())
        n_g002 = int((quant_df["group_nr"] == "G002").sum())
        up = int((stats_df["log2_fold_change"] > 0).sum())
        down = int((stats_df["log2_fold_change"] < 0).sum())

        summary = {
            "total_regions_analyzed": len(stats_df),
            "significant_corrected_p005": sig_corrected,
            "significant_uncorrected_p005": sig_uncorrected,
            "regions_upregulated_in_semaglutide": up,
            "regions_downregulated_in_semaglutide": down,
            "animals_vehicle_G001": n_g001,
            "animals_semaglutide_G002": n_g002,
            "marker": "c-Fos (proxy for neuronal activation)",
            "experiment": "Semaglutide (Ozempic/Wegovy active ingredient) vs Vehicle in mouse brain"
        }
        return json.dumps(summary, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a neuroscience research assistant analyzing brain imaging data from a mouse experiment.

The experiment compares two groups:
- G001 (Vehicle): control mice
- G002 (Semaglutide): mice treated with semaglutide (the active ingredient in Ozempic/Wegovy)

The marker is c-Fos — a protein expressed by recently active neurons, used as a proxy for neuronal activation.
The data covers hundreds of brain regions at single-region resolution.

You have access to tools to query the actual experimental data. When users ask about:
- Specific regions → use search_regions
- Top/most affected regions → use get_top_regions
- Overview questions → use get_summary_stats

After retrieving data, combine it with your knowledge of neuroanatomy to give insightful answers.
For example, if NTS (nucleus tractus solitarius) shows high activation, you know that region handles
satiety signals — so connect the dots to semaglutide's known mechanism of action.

Be concise and scientifically accurate. Use log2 fold change and corrected p-values when citing statistics.
Positive log2FC = more activation in semaglutide mice. Negative = less activation."""


# ── Agentic turn — one user message in, one full assistant response out ──────
def run_agent_turn(
    client,
    messages: list,
    stats_df: pd.DataFrame,
    quant_df: pd.DataFrame,
) -> str:
    """
    Run one full agent turn against the latest user message in `messages`.

    The function loops until Claude stops calling tools and returns plain text.
    `messages` is mutated in-place — assistant responses and intermediate tool
    results are appended so subsequent turns have the full conversation context.

    Args:
        client:    anthropic.Anthropic() instance
        messages:  conversation history; last entry must be the new user message
        stats_df:  statistics DataFrame (for tool execution)
        quant_df:  quantification DataFrame (for tool execution)

    Returns:
        The assistant's final text response (concatenated across any text blocks).
    """
    collected_text = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text blocks Claude emitted this round
        for block in response.content:
            if block.type == "text":
                collected_text.append(block.text)

        # End of turn — append final assistant message and return
        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            return "".join(collected_text)

        # Claude wants to call a tool — execute it and feed the result back
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input, stats_df, quant_df)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop_reason (e.g. max_tokens) — bail with what we have
        messages.append({"role": "assistant", "content": response.content})
        return "".join(collected_text)


# ── Interactive CLI chat (uses run_agent_turn under the hood) ────────────────
def chat(stats_df: pd.DataFrame, quant_df: pd.DataFrame):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    messages = []

    print("\n" + "="*60)
    print("  Brain Data Agent — Challenge B")
    print("  Powered by Claude — using your Anthropic API credits")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        response_text = run_agent_turn(client, messages, stats_df, quant_df)
        print(f"\nClaude: {response_text}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        print("Get your key from console.anthropic.com")
        exit(1)

    stats_df, quant_df = load_data()

    # Run a self-test to confirm tools work before interactive session
    print("\nSelf-test: querying top 3 upregulated regions...")
    result = execute_tool("get_top_regions", {"direction": "up", "n": 3, "significant_only": True}, stats_df, quant_df)
    print(result[:400])

    chat(stats_df, quant_df)
