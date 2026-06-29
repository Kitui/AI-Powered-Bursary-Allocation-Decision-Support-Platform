# AI-Powered Bursary Allocation Decision Support Platform

An AI-assisted bursary allocation platform for scoring applicant financial need, allocating limited bursary funds, and supporting transparent administrative review.

The system combines supervised machine learning with a DDPG reinforcement learning policy to predict financial need and distribute funds under a fixed budget. It includes CSV batch allocation, single applicant allocation, dashboard visualizations, allocation history, downloadable Excel reports, data validation, and allocation reason explanations.

## Core Stack

- Backend: Flask
- Database: SQLite with Flask-SQLAlchemy
- Authentication: Flask-Login and Flask-Bcrypt
- Machine learning: scikit-learn, XGBoost, LightGBM, Ridge stacking
- Reinforcement learning: PyTorch DDPG actor/critic
- Data processing: pandas, NumPy, PCA, sklearn preprocessing
- Frontend: Jinja templates, Bootstrap, Chart.js, DataTables
- Testing: pytest
- Deployment: Dockerfile, Flask on port 8080

## Key Features

- Batch CSV bursary allocation
- Single applicant allocation
- Financial need score prediction
- 7-feature DDPG allocation state:
  - predicted financial need score
  - vulnerability score
  - amount applied
  - PC1
  - PC2
  - PC3
  - PC4
- Budget-aware allocation capping
- Fairness metrics by gender, academic level, ward, and need score
- Allocation history with downloadable result files
- Excel export with allocation reasons
- CSV validation with structured error reporting
- Repeatable DDPG policy refresh using real applicant data

## Project Structure

```text
app.py                         Flask application
Analysis and Modelling.ipynb   Research and model training notebook
retrain_ddpg_policy.py         Refreshes DDPG actor/critic from Bursary.csv
requirements.txt               Python dependencies
Dockerfile                     Container build
templates/                     HTML pages
static/                        Static assets
tests/                         pytest contract tests
*.joblib / *.pth               Saved ML and RL artifacts
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:8080
```

## Testing

```bash
.venv\Scripts\python.exe -m pytest
```

## Model Refresh

To regenerate the 7-feature DDPG policy artifacts from the real bursary dataset:

```bash
.venv\Scripts\python.exe retrain_ddpg_policy.py
```

This updates:

- `ddpg_actor.pth`
- `ddpg_critic.pth`
- `ddpg_agent_params.joblib`

## Notes

Generated allocation Excel files, logs, local databases, virtual environments, and local Codex metadata are ignored by Git.
