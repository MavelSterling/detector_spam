from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable
import warnings

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE`.*", category=FutureWarning)

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "Goodmotion/spam-mail-classifier"
LABELS = ["NOSPAM", "SPAM"]


@lru_cache(maxsize=1)
def load_model():
    """Load the tokenizer and model once for the whole application."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()
    return tokenizer, model


def classify_texts(texts: Iterable[str]):
    """Classify one or more texts and return label, confidence, and scores."""
    clean_texts = [text.strip() for text in texts if text and text.strip()]
    if not clean_texts:
        return []

    tokenizer, model = load_model()
    inputs = tokenizer(
        clean_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    results = []

    labels = _model_labels(model)

    for text, probability in zip(clean_texts, probabilities):
        predicted_index = int(torch.argmax(probability).item())
        scores = {
            label: round(float(score), 4)
            for label, score in zip(labels, probability.tolist())
        }
        results.append(
            {
                "text": text,
                "label": labels[predicted_index],
                "confidence": round(float(probability[predicted_index].item()), 4),
                "scores": scores,
            }
        )

    return results


def classify_text(text):
    """Classify a single text."""
    results = classify_texts([text])
    return results[0] if results else None


def evaluate_messages(dataframe: pd.DataFrame, batch_size: int = 32) -> dict:
    """Evaluate the Hugging Face model on a dataframe with label/message columns."""
    if dataframe.empty:
        return {
            "accuracy": 0.0,
            "count": 0,
            "confusion_matrix": [],
            "classification_report": {},
            "predictions": [],
        }

    predictions = []
    messages = dataframe["message"].astype(str).tolist()
    for start in range(0, len(messages), batch_size):
        predictions.extend(classify_texts(messages[start : start + batch_size]))

    y_true = dataframe["label"].astype(str).str.upper().tolist()
    y_pred = [item["label"] for item in predictions]

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "count": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "predictions": predictions,
    }


def _model_labels(model):
    """Return model labels, falling back to the published mapping."""
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        return LABELS

    labels = [id2label.get(index, LABELS[index]) for index in range(len(LABELS))]
    labels = [_normalize_label(label) for label in labels]

    if set(labels) == {"LABEL_0", "LABEL_1"}:
        return LABELS

    return labels


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().upper().replace(" ", "")
    if normalized in {"HAM", "NOTSPAM", "NO-SPAM", "NONSPAM", "NOSPAM"}:
        return "NOSPAM"
    if normalized in {"SPAM", "JUNK"}:
        return "SPAM"
    return normalized
