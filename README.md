# Industrial AI — Predictive Maintenance for Energy Systems

End-to-end machine learning pipeline for anomaly detection and remaining useful life (RUL) prediction on industrial time-series data. Built as a prototype for energy infrastructure monitoring.

## Project structure
├── notebooks/        # Exploratory analysis and experiments
├── src/
│   ├── features/     # Feature engineering pipelines
│   ├── models/       # Model training and evaluation
│   └── visualization/# Dashboard and plotting utilities
├── data/
│   ├── raw/          # Original datasets (SKAB, NASA CMAPSS)
│   └── processed/    # Cleaned and engineered features
└── tests/            # Unit tests

## Datasets
- **SKAB** — Skoltech Anomaly Benchmark: hydraulic system sensor data with labeled anomalies
- **NASA CMAPSS** — Turbofan engine degradation simulation for RUL prediction

## Models
- Isolation Forest (unsupervised baseline)
- LSTM Autoencoder (reconstruction-based anomaly detection)
- LSTM RUL predictor (supervised degradation modeling)

## Key results
> To be updated as experiments run

## Stack
Python · TensorFlow · Scikit-learn · Streamlit · Docker
