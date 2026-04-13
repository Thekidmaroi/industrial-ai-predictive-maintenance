# Industrial AI — Predictive Maintenance for Energy Systems

End-to-end machine learning pipeline for anomaly detection and remaining useful life (RUL) prediction on industrial time-series data. Built as a prototype for energy infrastructure monitoring.

## Overview

This project implements a complete predictive maintenance system for industrial equipment, targeting energy infrastructure (pumps, valves, turbines). It combines classical machine learning and deep learning approaches to detect anomalies in SCADA sensor data before they lead to equipment failure.

**Context:** Developed as a prototype for the ERANOVE Group's operational technology stack, with the goal of reducing unplanned downtime and optimizing maintenance strategies across thermal power plants.

## Results

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Isolation Forest | 0.728 | 0.866 | 0.471 | 0.610 |
| LSTM Autoencoder | 0.589 | 0.956 | 0.358 | 0.521 |

**Key insight:** The two models have complementary profiles. Isolation Forest offers better global discrimination (AUC 0.73) while the LSTM Autoencoder achieves higher anomaly recall (0.96). A production ensemble would combine both — flagging an alert when either model exceeds its threshold.

## Project Structure
├── notebooks/
│   ├── 01_skab_exploration.ipynb       # EDA and sensor analysis
│   ├── 02_isolation_forest.ipynb       # Baseline anomaly detection
│   └── 03_lstm_autoencoder.ipynb       # Deep learning approach
├── src/
│   ├── features/
│   │   └── engineering.py             # Data loading and feature engineering
│   ├── models/
│   │   ├── isolation_forest.py        # IF detector with threshold optimization
│   │   ├── lstm_autoencoder.py        # LSTM AE architecture
│   │   └── evaluator.py              # Metrics and model comparison
│   └── visualization/
│       └── plots.py                  # Plotting utilities
├── tests/                            # Unit tests (12 tests, 100% passing)
├── app_dashboard.py                  # Interactive Streamlit dashboard
├── config.yaml                       # Centralized configuration
└── Dockerfile                        # Container deployment

## Datasets

**SKAB — Skoltech Anomaly Benchmark**
- Hydraulic system sensor data (8 sensors, 1Hz sampling)
- 37,401 data points across valve1, valve2, other scenarios
- 34.9% labeled anomalies

**NASA CMAPSS** *(Phase 2 — in progress)*
- Turbofan engine degradation simulation
- Remaining Useful Life (RUL) prediction

## Technical Stack

| Layer | Technology |
|---|---|
| Data processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| MLOps | Docker, GitHub Actions |
| Testing | Pytest (12 unit tests) |

## Running Locally

**With Python:**
```bash
pip install -r requirements.txt
streamlit run app_dashboard.py --server.port 8502
```

**With Docker:**
```bash
docker build -t industrial-ai .
docker run -p 8502:8502 industrial-ai
```

## Key Technical Decisions

**Why train on normal data only?**
In industrial settings, labeled anomalies are rare and expensive to collect. Training the Isolation Forest and LSTM Autoencoder exclusively on normal behavior allows deployment on new equipment with no historical failures.

**Why prioritize Recall over Precision?**
In predictive maintenance, a missed failure (false negative) costs far more than a false alarm (false positive). The threshold optimization explicitly maximizes Recall while maintaining acceptable Precision.

**Why rolling features?**
Raw sensor values have low discriminative power (delta < 5% for most sensors). Rolling standard deviation over 30-second windows captures the variance increase that characterizes anomalies — Accelerometer1RMS variance doubles during anomaly periods.

## CI/CD

GitHub Actions pipeline runs on every push to main:
- Unit tests (pytest)
- Syntax validation
- Dependency installation check

## Author

Marwane Houngnon — AI Research Engineer
Published author (Springer Nature, Web of Science indexed)
Research focus: hybrid probabilistic-neural architectures for energy systems
