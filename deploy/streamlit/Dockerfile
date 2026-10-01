# Streamlit dashboard image — a read-only thin client over src/ on the NAS.
# Reads the warehouse over the LAN and the deployed model from model_registry
# (FACTOR_MODEL_STORE=db), so it needs no model volume. constraints-ml.txt pins
# the ML stack to the training image so the registry artifact unpickles here.
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime LightGBM needs to score the deployed model.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt constraints-ml.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt -c constraints-ml.txt

COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/

ENV PYTHONUNBUFFERED=1 FACTOR_MODEL_STORE=db
EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
