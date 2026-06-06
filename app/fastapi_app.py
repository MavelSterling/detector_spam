from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.data import DATASET_SOURCE, dataset_sample, dataset_summary
from app.model import MODEL_NAME, classify_text, classify_texts


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Urgent: Verify your account immediately."])


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        examples=[["Meeting rescheduled to 3 PM", "Claim your free prize now!"]],
    )


api = FastAPI(
    title="Spam Mail Detector API",
    description="API para clasificar mensajes como SPAM o NOSPAM usando Goodmotion/spam-mail-classifier.",
    version="1.0.0",
)


@api.get("/", response_class=HTMLResponse)
def web_ui():
    return HTMLResponse(_index_html())


@api.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dataset": DATASET_SOURCE,
    }


@api.post("/predict")
def predict(payload: PredictRequest):
    result = classify_text(payload.text)
    if result is None:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacio.")
    return result


@api.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest):
    results = classify_texts(payload.texts)
    if not results:
        raise HTTPException(status_code=400, detail="Incluye al menos un texto valido.")
    return {"count": len(results), "results": results}


@api.get("/dataset")
def dataset(limit: Annotated[int, Query(ge=1, le=100)] = 10):
    return {
        "summary": dataset_summary(),
        "sample": dataset_sample(limit),
    }


def _index_html() -> str:
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spam Mail Detector</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f6f7fb; color: #172033; }
    main { width: min(880px, calc(100% - 32px)); margin: 40px auto; }
    h1 { margin: 0 0 8px; font-size: clamp(2rem, 5vw, 3.4rem); }
    p { line-height: 1.5; color: #586174; }
    section { background: white; border: 1px solid #dde2ea; border-radius: 8px; padding: 20px; margin-top: 18px; }
    textarea { width: 100%; min-height: 130px; resize: vertical; box-sizing: border-box; padding: 12px; border: 1px solid #bcc5d3; border-radius: 6px; font: inherit; }
    button { margin-top: 12px; border: 0; border-radius: 6px; padding: 10px 14px; background: #1769e0; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #111827; color: #e5e7eb; border-radius: 6px; padding: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>Spam Mail Detector</h1>
    <p>Clasificador de mensajes con el modelo Goodmotion/spam-mail-classifier y dataset UCI SMS Spam Collection.</p>
    <section>
      <label for="message"><strong>Mensaje</strong></label>
      <textarea id="message">Urgent: Verify your account immediately.</textarea>
      <button id="predict">Clasificar</button>
      <pre id="result">Esperando mensaje...</pre>
    </section>
  </main>
  <script>
    const button = document.querySelector("#predict");
    const output = document.querySelector("#result");
    button.addEventListener("click", async () => {
      button.disabled = true;
      output.textContent = "Clasificando...";
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: document.querySelector("#message").value })
        });
        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        output.textContent = "Error: " + error.message;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
