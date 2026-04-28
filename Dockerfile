FROM python:3.12-slim

WORKDIR /app

# copy deps first so pip layer is cached on code-only changes
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the API source and model artifacts
COPY api/ ./api/
COPY satellite-damage-detection/models/ ./satellite-damage-detection/models/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]