from pathlib import Path

# ── Nebius Token Factory models ───────────────────────────────────────────────
# See cc_docs/model_selection.md for full justification.
GENERATION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"      # agent ReAct loop + summarization
ROUTER_MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # router + profile updater (Task 2)
NEBIUS_BASE_URL  = "https://api.tokenfactory.nebius.com/v1/"

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_ITERATIONS        = 12   # graceful fallback after this many ReAct steps
SUMMARIZE_SAMPLE_SIZE = 20   # rows sampled inside summarize_subset

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
