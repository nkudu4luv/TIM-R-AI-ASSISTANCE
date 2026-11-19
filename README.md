# TIM-R Academy Handbook Assistant

A small project that answers questions about the TIM-R Academy school handbook using a lightweight retrieval workflow. It supports both:

- a command-line assistant via `main.py`
- a Streamlit web app via `app.py`

## Setup

Use Python 3.10 or later. From the project folder, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For OpenAI-generated answers, create a `.env` file in the project folder:

```text
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

The app also runs without an API key using its local fallback mode.

## Project structure

- `main.py` — terminal-based chat assistant
- `app.py` — Streamlit web UI
- `handbook.py` — handbook loading and chunking logic
- `retriever.py` — embedding, retrieval, and answer generation logic
- `.gitignore` — excludes virtual environments, secrets, and generated files

## Run the CLI assistant

```bash
python main.py
```

## Run the Streamlit app

```bash
streamlit run app.py
```

## Notes

- Place a handbook at `documents/school_handbook.txt` or `school_handbook.txt`. If neither file is found, the app uses the built-in handbook.
- If the OpenAI API is unavailable or rate-limited, the app uses a local fallback answer mode so it still works.
