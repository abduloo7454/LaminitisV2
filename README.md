# LaminitisCare (Streamlit)

Laminitis risk assistant for single-horse predictions with guidance and an agent assistant.

## Local run

```bash
python3 -m venv .laminitis-venv
. .laminitis-venv/bin/activate
python -m pip install -r requirements.txt
streamlit run Streamlit/main.py
```

Place these files at repo root (already included here):
- svm_model.pkl
- svm_scaler.pkl (not used when domain scaling is active)
- svm_features.pkl

Optional (agentic AI): add `Streamlit/openai_api_key.txt` containing your key, or set `OPENAI_API_KEY`.

## Deploy to Streamlit Cloud
- Push this repo to GitHub.
- On Streamlit Cloud, set the app entrypoint to `Streamlit/main.py`.
- Add `OPENAI_API_KEY` as a secret if you want LLM agent; otherwise it runs local agent only.

## Notes
- The app enforces correct types and ranges; unknown columns are ignored; missing features imputed.
- Predictions use domain-based scaling per specified feature ranges.
