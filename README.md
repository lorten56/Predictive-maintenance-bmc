# Predictive Maintenance Digital Twin — Streamlit Prototype

An interactive Streamlit dashboard built from the supplied **synthetic** 2021–2025 predictive maintenance starter CSV data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload **the contents of this folder** to the root of the repository, keeping the `data/` folder and its CSV files.
3. Visit https://share.streamlit.io/ and choose **Create app** / **Deploy an app**.
4. Select your repository, branch `main`, and entrypoint `app.py`.
5. Deploy and wait for the dependencies to install and the model to train on startup.

The model trains on 2021–2023, chooses a threshold on 2024 validation data, and reports 2025 holdout metrics. Risk predictions shown are historical test-period estimates. This is **not** a validated live prediction system. No confidential operational records should be added to a public repository.
