# Use a specific platform and Python version for consistency
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including Tesseract and OpenCV's dependency
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# --- CRITICAL STEP: Download and cache the model during the build ---
# This ensures the final image is self-contained and can run offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy the rest of your application code
COPY . .

# Command to run your relevance analyzer script
CMD ["python", "relevance_analyzer.py"]
