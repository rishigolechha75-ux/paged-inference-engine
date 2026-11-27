FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install --no-cache-dir -r requirements.txt

COPY pagedserve ./pagedserve
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
# GPT-2 weights (~548 MB) are downloaded from the Hugging Face Hub on first start.
CMD ["pagedserve", "serve", "--host", "0.0.0.0", "--port", "8000"]
