# Persona-Driven Document Relevance Analyzer (Round 1B)

## 1. Project Overview

This project provides a complete, dockerized pipeline to automatically analyze collections of PDF documents. It identifies and ranks the most relevant sections based on a given user persona and their specific "job-to-be-done."

The solution uses a sophisticated two-stage process that combines the structural understanding from a machine learning classifier (developed in Round 1A) with the semantic understanding of a modern Sentence Transformer model. This hybrid approach allows for the extraction of highly relevant and contextually aware results, even from complex and scanned documents.

## 2. How It Works: A Hybrid Two-Stage Pipeline

The intelligence of this system comes from its ability to understand both *what* a piece of text is (its structure) and *what it means* (its semantics).

### Stage 1: Structural Analysis (Using 1A Model)

The script first processes each PDF to understand its layout and structure.

* **Hybrid PDF Parsing:** A robust parser handles both text-based and image-based (scanned) PDFs.
    * It first attempts fast, native text extraction.
    * If a page has little selectable text, it automatically falls back to an OCR engine (Tesseract).
    * **OCR Pre-processing:** Before OCR, images are cleaned using OpenCV (grayscaling, adaptive thresholding) to maximize accuracy.
* **Feature Engineering:** The script engineers a rich set of features for each line of text, including font size, boldness, position, and relative font size compared to the rest of the page.
* **Structure Prediction:** It uses the pre-trained Random Forest model (`document_classifier.joblib` from Round 1A) to predict the structural role of each line (e.g., `H1`, `H2`, `H3`, `O` for body text).

### Stage 2: Semantic Relevance Analysis (Using 1B Model)

With the document's structure understood, the script then determines the relevance of its content.

* **Intelligent Chunking:** Instead of treating each page as a blob of text, the script intelligently groups the content into semantic sections. A section is defined as a predicted heading (`H1`, `H2`, etc.) plus all the body text that follows it, up to the next heading. This creates contextually rich chunks for analysis.
* **Semantic Embeddings:** It uses a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) to convert text into numerical vectors ("embeddings") that represent its meaning. This is done for:
    1.  The user's goal (a combination of the `persona` and `job_to_be_done`).
    2.  Each of the semantic chunks extracted from the documents.
* **Relevance Ranking:** The script calculates the cosine similarity between the user's goal embedding and each chunk's embedding. This score is further enhanced with boosts based on keyword matches and the importance of the heading level (e.g., `H1` content is weighted more heavily).
* **Output Generation:** The chunks are ranked by their final relevance score, and the top results are formatted into the required JSON output, including the actual heading text as the `section_title`.

## 3. Directory Structure

To run the project, you **must** organize your files in the following structure. The script will automatically discover and process each `Collection` folder.

```
/your_project_folder/
|
|-- Collection 1/
|   |-- PDFs/
|   |   |-- doc1.pdf
|   |-- challenge1b_input.json
|
|-- Collection 2/
|   |-- PDFs/
|   |   |-- docA.pdf
|   |-- challenge1b_input.json
|
|-- model_1a/  <-- IMPORTANT: Contains the models from Round 1A
|   |-- document_classifier.joblib
|   |-- label_encoder.joblib
|
|-- relevance_analyzer.py
|-- requirements.txt
|-- Dockerfile
```

## 4. Prerequisites

* **Docker Desktop:** The entire application is containerized, so you only need Docker installed and running on your machine.

## 5. Steps of Execution

Follow these steps from your terminal inside `your_project_folder`.

### Step 1: Build the Docker Image

This command packages your script and all its dependencies—including the Sentence Transformer AI model and NLTK tokenizer—into a self-contained image named `relevance-analyzer`. This may take a few minutes on the first run.

```bash
docker build -t relevance-analyzer .
```

### Step 2: Run the Analyzer

This command runs the script. It mounts your entire project folder into the container, allowing the script to find the `Collection` folders and write the output files directly back to them.

```bash
docker run --rm -v "./:/app" relevance-analyzer
```

### Step 3: Check Your Results

That's it! The script will process all collections, and a `challenge1b_output.json` file will be created or updated directly inside each of your local `Collection 1`, `Collection 2`, etc., folders. **No manual file copying is needed.**

## 6. Output Format

The `challenge1b_output.json` file generated in each collection folder will have the following structure:

```json
{
    "metadata": {
        "input_documents": [
            "doc1.pdf",
            "doc2.pdf"
        ],
        "persona": "Travel Planner",
        "job_to_be_done": "Plan a trip of 4 days...",
        "processing_timestamp": "2025-07-26T20:30:00.123456"
    },
    "extracted_sections": [
        {
            "document": "doc1.pdf",
            "page_number": 5,
            "section_title": "Coastal Adventures",
            "importance_rank": 1
        }
    ],
    "subsection_analysis": [
        {
            "document": "doc1.pdf",
            "page_number": 5,
            "refined_text": "The South of France is renowned for its beautiful coastline..."
        }
    ]
}
