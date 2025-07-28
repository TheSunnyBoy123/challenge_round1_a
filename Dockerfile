# Step 1: Specify the base image, ensuring AMD64 compatibility
FROM --platform=linux/amd64 python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies required by your script
# - poppler-utils is for pdf2image to convert PDFs
# - tesseract-ocr is the OCR engine for pytesseract
# - libgl1 is a dependency for OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install Python packages
# Using --no-cache-dir keeps the image size smaller
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your script and model file into the container's working directory
COPY final.py .
COPY yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt .

# Step 6: Define the command to run when the container starts
CMD ["python", "final.py"]
