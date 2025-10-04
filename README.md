# DeepData Flood Risk Dashboard

This repository contains a Dash-based dashboard to visualize urban flood risk hotspots.

Quick start (local):

1. Create and activate a virtual environment (PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python dashboard.py
```

2. Open http://127.0.0.1:8050/ in your browser.

Files to push to GitHub (recommended)

Include these project files in your repository when pushing to GitHub (Render will use them to build and run the app):

- `dashboard.py` (the Dash app)
- `requirements.txt` (Python dependencies)
- `render.yaml` (Render manifest)
- `README.md`
- `.gitignore`

Do NOT commit large or sensitive data files like `flood_risk_dataset.csv`. Keep those outside the repo and load remotely, or store them in a private storage bucket and use environment variables for credentials.

Render deployment (recommended for Dash apps)

1. Sign up / log in to Render and connect your GitHub repository: https://render.com
2. Create a new Web Service from your repository and choose the `main` branch (or the branch you want to deploy).
3. Use the following settings:
	- Environment: Python
	- Build Command: pip install -r requirements.txt
	- Start Command: gunicorn dashboard:app
	- Plan: Free (or select the plan you prefer)
4. Alternatively, Render will auto-detect a `render.yaml` manifest in the repo — one is included in this project. The manifest specifies a free Python web service named `deepdata-dashboard`.
5. After creating the service, Render will build and deploy and provide a public URL.

Notes about the free plan:
- Free services may sleep when idle and have resource limits; check Render's current policy.
- For persistent production usage consider upgrading the plan or hosting data externally.
