# Spam Mail Detector

Spam Mail Detector is a Python web application that classifies SMS, email subjects, or short messages as `SPAM` or `NOSPAM`.

The project uses the Hugging Face model [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier), a multilingual text-classification model fine-tuned from `microsoft/Multilingual-MiniLM-L12-H384` for spam detection. It also inspects the UCI [`SMS Spam Collection`](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip) dataset when the file is available locally.

## Project Structure

```text
detector_spam/
|-- app/
|   |-- __init__.py
|   |-- data.py
|   |-- fastapi_app.py
|   |-- model.py
|   `-- streamlit_app.py
|-- data/
|   `-- SMSSpamCollection
|-- notebook/
|   |-- requirements.txt
|   `-- spam_mail_classifier_process.ipynb
|-- README.md
`-- requirements.txt
```

## Features

- Streamlit frontend organized into three tabs:
  - Exploratory data analysis for the UCI SMS Spam Collection.
  - Model evaluation results with accuracy, precision, recall, F1-score, confusion matrix, and prediction examples.
  - Chat-style message classification.
- Classifies a single email subject or message.
- Classifies multiple examples in a batch.
- Shows the predicted label and confidence score.
- Uses a pretrained Transformer model from Hugging Face.
- Exposes a FastAPI service with prediction and dataset endpoints.
- Reads the local UCI SMS Spam Collection file for dataset summaries, charts, samples, and model evaluation.
- Keeps notebook-only dependencies outside the Docker runtime requirements.

## Installation

Follow these steps to run the project locally.

1. Clone or open the project folder:

```powershell
cd detector_spam
```

2. Create a Python virtual environment inside the project:

```bash
python -m venv .venv
```

3. Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Make sure the dataset file exists at:

```text
data/SMSSpamCollection
```

To run the notebook, install its additional dependencies from the notebook folder:

```powershell
pip install -r notebook/requirements.txt
```

The Hugging Face model is downloaded on demand the first time the app needs it. The dataset is not downloaded by the app; it must be present in `data/SMSSpamCollection`.

## Run The Web App

### Local Streamlit App

Run the main frontend:

```bash
streamlit run app/streamlit_app.py
```

Then open:

- Streamlit app: http://localhost:8501

Windows PowerShell fallback if the file watcher closes the local server:

```powershell
python -m streamlit run app/streamlit_app.py --server.fileWatcherType none
```

### Local FastAPI Service

Run the API:

```bash
uvicorn app.fastapi_app:api --reload
```

Then open:

- Web UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## Run With Docker

The Docker image installs the app dependencies from the root `requirements.txt`. PyTorch is pinned to the CPU wheel with `torch==2.5.1+cpu` to avoid installing GPU/CUDA packages.

Build the image:

```bash
docker build -t detector-spam .
```

Run the Streamlit app:

```bash
docker run --rm -p 8501:8501 detector-spam
```

Then open:

- Streamlit app: http://localhost:8501

The dataset file is copied into the image because it is small. The Hugging Face model is still downloaded on demand the first time the container needs it.

## FastAPI Endpoints

- `GET /health`: service status, model, and dataset source.
- `POST /predict`: classify one text.
- `POST /predict/batch`: classify multiple texts.
- `GET /dataset?limit=10`: dataset summary and sample rows.

## Data Science Process

The notebook in `notebook/spam_mail_classifier_process.ipynb` describes the project workflow:

1. Problem definition.
2. Model reference and selection.
3. Environment setup.
4. Text preprocessing considerations.
5. Single-message inference.
6. Batch inference.
7. Result interpretation.
8. Web application deployment with Streamlit and FastAPI.

## Reference

- Hugging Face model: [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier)
- UCI dataset: [`SMS Spam Collection`](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)
