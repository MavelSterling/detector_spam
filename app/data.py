from functools import lru_cache
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_FILE = DATA_DIR / "SMSSpamCollection"
DATASET_SOURCE = str(RAW_FILE)


def ensure_dataset() -> Path:
    """Return the local UCI SMS Spam Collection dataset file."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_FILE}. Place the SMSSpamCollection file in {DATA_DIR}."
        )

    return RAW_FILE


@lru_cache(maxsize=1)
def load_sms_dataset() -> pd.DataFrame:
    """Load the SMS Spam Collection as a normalized dataframe."""
    raw_file = ensure_dataset()
    dataframe = pd.read_csv(
        raw_file,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )
    dataframe["label"] = dataframe["label"].str.upper().replace({"HAM": "NOSPAM"})
    dataframe["message"] = dataframe["message"].astype(str)
    return dataframe


def dataset_summary() -> dict:
    """Return compact dataset metadata for UI and API clients."""
    dataframe = load_sms_dataset()
    counts = dataframe["label"].value_counts().to_dict()
    enriched = _with_text_features(dataframe)
    spam_count = int(counts.get("SPAM", 0))
    nospam_count = int(counts.get("NOSPAM", 0))
    return {
        "source": DATASET_SOURCE,
        "rows": int(len(dataframe)),
        "labels": {label: int(count) for label, count in counts.items()},
        "spam_rate": round(spam_count / len(dataframe), 4),
        "nospam_rate": round(nospam_count / len(dataframe), 4),
        "avg_characters": round(float(enriched["characters"].mean()), 2),
        "avg_words": round(float(enriched["words"].mean()), 2),
        "max_characters": int(enriched["characters"].max()),
    }


def dataset_sample(limit: int = 10) -> list[dict]:
    """Return a deterministic sample of dataset rows."""
    dataframe = load_sms_dataset().head(max(1, min(limit, 100)))
    return dataframe.to_dict(orient="records")


def dataset_profile() -> dict:
    """Return dataframes for exploratory charts in the frontend."""
    dataframe = _with_text_features(load_sms_dataset())

    label_distribution = (
        dataframe["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    length_by_label = dataframe.groupby("label", as_index=False).agg(
        avg_characters=("characters", "mean"),
        median_characters=("characters", "median"),
        avg_words=("words", "mean"),
        median_words=("words", "median"),
    )
    length_by_label = length_by_label.round(2)
    common_terms = _common_terms(dataframe, limit=15)

    return {
        "dataframe": dataframe,
        "label_distribution": label_distribution,
        "length_by_label": length_by_label,
        "common_terms": common_terms,
    }


def evaluation_sample(limit: int = 300) -> pd.DataFrame:
    """Return a deterministic stratified sample for model evaluation."""
    dataframe = load_sms_dataset()
    limit = max(2, min(limit, len(dataframe)))
    per_label = max(1, limit // dataframe["label"].nunique())
    sample = (
        dataframe.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), per_label), random_state=42))
        .sample(frac=1, random_state=42)
    )

    if len(sample) < limit:
        remainder = dataframe.drop(sample.index).sample(
            min(limit - len(sample), len(dataframe) - len(sample)),
            random_state=42,
        )
        sample = pd.concat([sample, remainder])

    return sample.sample(frac=1, random_state=42).head(limit).reset_index(drop=True)


def _with_text_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    featured = dataframe.copy()
    featured["characters"] = featured["message"].str.len()
    featured["words"] = featured["message"].str.split().str.len()
    return featured


def _common_terms(dataframe: pd.DataFrame, limit: int) -> pd.DataFrame:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "u",
        "ur",
        "with",
        "you",
        "your",
    }
    terms = (
        dataframe["message"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.split()
        .explode()
    )
    terms = terms[terms.str.len().gt(2) & ~terms.isin(stop_words)]
    return terms.value_counts().head(limit).rename_axis("term").reset_index(name="count")
