import pandas as pd
from functools import lru_cache

from datasets import load_dataset

from agent.config import DATA_DIR, DATASET_NAME


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    """Load the Bitext customer-service dataset and return it as a cached DataFrame.

    The HuggingFace datasets library caches the raw download to DATA_DIR;
    subsequent calls return the in-process cached DataFrame instantly.
    """
    ds = load_dataset(DATASET_NAME, split="train", cache_dir=str(DATA_DIR))
    return ds.to_pandas()


def describe() -> None:
    """Print a schema summary — run once to learn column names and valid category/intent values."""
    df = load()
    print(f"Shape:   {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    cats = sorted(df["category"].unique())
    print(f"Categories ({len(cats)}): {cats}\n")
    for cat in cats:
        intents = sorted(df[df["category"] == cat]["intent"].unique())
        print(f"  {cat} ({len(intents)} intents): {intents}")
