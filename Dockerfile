FROM python:3.12-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    tensorflow-cpu \
    streamlit==1.35.0 \
    plotly==5.22.0 \
    pyyaml==6.0.1

# Code source
COPY src/ ./src/
COPY app_dashboard.py .
COPY config.yaml .

EXPOSE 8502

CMD ["streamlit", "run", "app_dashboard.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
