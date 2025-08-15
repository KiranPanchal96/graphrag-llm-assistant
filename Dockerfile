FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and local model
COPY . .
COPY my_local_model /app/my_local_model
ENV SENTENCE_TRANSFORMERS_HOME=/app/my_local_model

EXPOSE 8000

# Use uvicorn to serve FastAPI 
CMD ["uvicorn", "src.api.s09a_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
