FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_all.py .
COPY src/ src/

RUN mkdir -p data results paper_sections

ENTRYPOINT ["python", "run_all.py"]
