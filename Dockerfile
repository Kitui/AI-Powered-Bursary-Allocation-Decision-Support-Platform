# Use a stable Python 3.9 base image
FROM python:3.9

# Install system dependencies for LightGBM, XGBoost, SQLite, and general build tools
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
    make \
    cmake \
    libomp-dev \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY rf_model_tuned.joblib .
COPY xgb_model_tuned.joblib .
COPY lgb_model_tuned.joblib .
COPY meta_model_tuned.joblib .
COPY final_feature_names.joblib .
COPY ddpg_actor.pth .
COPY ddpg_critic.pth .
COPY ddpg_agent_params.joblib .
COPY pca.joblib .

# Create data directory for persistent storage
RUN mkdir -p /app/data && chmod -R 755 /app/data

# Set permissions for static files
RUN chmod -R 644 /app/static/*

# Define a volume for persistent data
VOLUME /app/data

# Expose the Flask port
EXPOSE 8080

# Run the Flask application
CMD ["python", "app.py"]
