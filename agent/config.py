from pathlib import Path

# ── Nebius Token Factory models ───────────────────────────────────────────────
# See cc_docs/model_selection.md for full justification.
GENERATION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"      # agent ReAct loop + summarization
ROUTER_MODEL     = "Qwen/Qwen3-30B-A3B-Instruct-2507"        # router + profile updater (Task 2) — MoE, 3B active params
NEBIUS_BASE_URL  = "https://api.tokenfactory.nebius.com/v1/"

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_ITERATIONS        = 12   # graceful fallback after this many ReAct steps
SUMMARIZE_SAMPLE_SIZE = 20   # rows sampled inside summarize_subset

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).parent.parent
DATA_DIR         = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHECKPOINTS_DB   = DATA_DIR / "checkpoints.sqlite"   # Task 2a episodic memory
PROFILES_DIR     = DATA_DIR / "profiles"              # Task 2b per-user profile files

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
