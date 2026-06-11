FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for building or running gTTS, pydub, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pyquaternion and nuscenes-devkit separately to avoid numpy 1.x vs 2.x compilation conflict
RUN pip install --no-cache-dir pyquaternion && \
    pip install --no-cache-dir nuscenes-devkit --no-deps

COPY . .

# Generate the initial evaluation metrics and mock datasets
RUN python data/extractor.py --test && \
    python eval/evaluate.py && \
    python paper/paper_generator.py

EXPOSE 8000

CMD ["python", "backend/main.py"]
