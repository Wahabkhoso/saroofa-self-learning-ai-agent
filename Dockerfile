FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal on purpose - no build toolchain needed for
# the wheels this project uses.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend

RUN mkdir -p /app/data/db /app/data/chroma /app/data/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
