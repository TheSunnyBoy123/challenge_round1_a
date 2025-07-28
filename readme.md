# Document Outline Extraction from PDF
This solution automatically extracts a structured hierarchical outline from PDF documents. It identifies the document's title and nested section headers (e.g., H1, H2, H3), generating a clean JSON output for each input file.

## Approach
The solution employs a sophisticated multi-stage pipeline designed for accuracy and robustness, processing each PDF from a read-only input directory and writing to a designated output directory. The entire process is designed to run in a temporary, isolated environment to ensure clean and predictable execution.

The pipeline operates in three main stages:

1. Pre-processing: PDF to Image Conversion
Since the core layout analysis model operates on visual data, the first step is to convert each page of an input PDF into a high-resolution (300 DPI) image. This ensures that all content, whether text-based or scanned, is available for analysis. This process uses a temporary directory inside the container to store the generated images, keeping the mounted input/output directories clean.

2. Layout Analysis and Hierarchy Inference
This is the core of the solution, where each document's images are processed to identify and structure its contents.

**Layout Detection:** A YOLOv8 object detection model, pre-trained on the DocLayNet dataset, is used to identify the bounding boxes of key layout components, specifically Title and Section-header elements on each page.

**Text Extraction:** For each bounding box detected by YOLO, the corresponding image region is cropped. Tesseract OCR is then applied to this small region to accurately extract the text content of the title or header.

**Hierarchical Analysis:** Once all headers are extracted, a hybrid rule-based and machine-learning model determines their structural level (h1, h2, etc.):

**Rule-Based Pass:** The system first scans header text for explicit numerical prefixes (e.g., "1. Introduction", "2.1. Methodology"). It uses these patterns to reliably assign initial hierarchy levels (e.g., "1." -> h1, "2.1." -> h2).

**ML-Based Pass (Style Analysis):** For headers without numerical prefixes, the system analyzes stylistic features like font size, indentation, and centering.

If numbered headers were found, it creates style profiles (centroids) for each level (h1, h2). Un-numbered headers are then assigned to the level with the most similar style.

If no numbered headers exist, it uses Agglomerative Clustering (from scikit-learn) to group headers by stylistic similarity. The clusters are then ranked by average font size to infer the hierarchy (larger fonts correspond to higher levels, e.g., h1).

**Intermediate Output:** The rich data from this stage (including text, bounding boxes, confidence scores, and assigned hierarchy levels) is saved to an intermediate JSON file in a temporary directory.

3. Final Output Formatting
In the final stage, the pipeline reads the intermediate layout file for each document and transforms it into the clean, required JSON format. It extracts the document title and compiles the final outline, ensuring the output is simple and conforms to the specified schema. The final JSON is named after the original PDF and placed in the /app/output directory.

## Models and Libraries
This solution is built entirely on open-source technologies.

### Models
**yolov8x-doclaynet:** A large YOLOv8 object detection model fine-tuned on the DocLayNet dataset. It is highly effective at identifying structural elements in a wide variety of document layouts. The model file is included in the Docker image and is expected to be at /app/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt.

### Key Python Libraries
**ultralytics:** The framework used to load and run the YOLOv8 model.

**pytesseract & opencv-python-headless:** Used together to perform Optical Character Recognition (OCR) on image regions to extract text.

**pdf2image:** A wrapper for the poppler utility to convert PDF files into PIL Image objects for processing.

**scikit-learn:** Provides the AgglomerativeClustering and StandardScaler for the style-based hierarchy analysis.

**Pillow (PIL):** Used for all image manipulation tasks.

**numpy:** A fundamental dependency for numerical operations, especially for feature matrices in the ML clustering step.

## System Dependencies
The Docker container environment requires the following system-level packages to be installed:

**tesseract-ocr:** The underlying OCR engine.

**poppler-utils:** The backend library required by pdf2image for PDF rendering.

## How to Build and Run
This solution is containerized using Docker and is designed to run in an isolated environment without network access.

1. Build the Docker Image
Navigate to the directory containing the Dockerfile and run the required `docker build` command.

This command builds a Docker image for the linux/amd64 architecture, installs all system and Python dependencies, and copies the model and script into the image.

2. Run the Container
To process your PDFs, place them in a local directory (e.g., input). Create an empty local directory for the results (e.g., output/repoidentifier). Then, run the container using the following command: