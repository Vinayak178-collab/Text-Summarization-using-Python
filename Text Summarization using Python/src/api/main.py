from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.preprocessing.pdf_loader import extract_text_from_pdf_bytes
from src.summarizers.abstractive import AbstractiveSummaryResult, abstractive_summary
from src.summarizers.extractive import ExtractiveSummaryResult, centroid_extractive_summary


app = FastAPI(title="Text Summarizer", version="0.1.0")


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Input text to summarize")
    mode: str = Field("extractive", description="extractive or abstractive")
    num_sentences: int = Field(3, description="Number of sentences for extractive summary")
    max_length: int = Field(130, description="Max tokens for abstractive summary")
    min_length: int = Field(30, description="Min tokens for abstractive summary")


class SummarizeResponse(BaseModel):
    summary: str
    mode: str
    details: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    mode = request.mode.lower()

    if mode == "extractive":
        result: ExtractiveSummaryResult = centroid_extractive_summary(
            text=text,
            num_sentences=request.num_sentences,
        )
        return SummarizeResponse(
            summary=result.summary,
            mode="extractive",
            details={
                "indices": result.indices,
                "scores": result.scores,
                "selected_sentences": result.selected_sentences,
            },
        )
    elif mode == "abstractive":
        result: AbstractiveSummaryResult = abstractive_summary(
            text=text,
            max_length=request.max_length,
            min_length=request.min_length,
        )
        return SummarizeResponse(
            summary=result.summary,
            mode="abstractive",
            details={
                "chunks": result.chunks,
                "chunk_summaries": result.chunk_summaries,
            },
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'extractive' or 'abstractive'.")


@app.post("/upload", response_model=SummarizeResponse)
async def upload_and_summarize(
    file: UploadFile = File(...),
    mode: str = Form("extractive"),
    num_sentences: int = Form(3),
    max_length: int = Form(130),
    min_length: int = Form(30),
) -> SummarizeResponse:
    """
    Upload a PDF file and summarize its contents.
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported for now.")

    data = await file.read()
    text = extract_text_from_pdf_bytes(data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

    request = SummarizeRequest(
        text=text,
        mode=mode,
        num_sentences=num_sentences,
        max_length=max_length,
        min_length=min_length,
    )
    return summarize(request)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """
    Minimal HTML UI for quick testing.
    """
    return """
    <!DOCTYPE html>
    <html lang="en" data-bs-theme="dark">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Summarizer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
        <style>
            :root {
                --primary-color: #6c63ff;
                --secondary-color: #4a45b1;
            }
            body {
                background-color: #1a1a2e;
                color: #e6e6e6;
                min-height: 100vh;
            }
            .card {
                background-color: #16213e;
                border: 1px solid #2a3a5c;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .btn-primary {
                background-color: var(--primary-color);
                border-color: var(--primary-color);
            }
            .btn-primary:hover {
                background-color: var(--secondary-color);
                border-color: var(--secondary-color);
            }
            .form-control, .form-select {
                background-color: #0f3460;
                border: 1px solid #2a3a5c;
                color: #e6e6e6;
            }
            .form-control:focus, .form-select:focus {
                background-color: #0f3460;
                color: #e6e6e6;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 0.25rem rgba(108, 99, 255, 0.25);
            }
            #summary {
                background-color: #0f3460;
                border: 1px solid #2a3a5c;
                border-radius: 5px;
                color: #e6e6e6;
                font-family: inherit;
                line-height: 1.6;
            }
            .header {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 2rem 0;
                margin-bottom: 2rem;
                border-bottom: 1px solid #2a3a5c;
            }
            .card-header {
                background-color: rgba(0, 0, 0, 0.1);
                border-bottom: 1px solid #2a3a5c;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
    <div class="header">
        <div class="container">
            <h1 class="display-5 fw-bold"><i class="bi bi-text-paragraph me-2"></i>Text Summarizer</h1>
            <p class="text-muted">Generate concise summaries of your text using AI</p>
        </div>
    </div>
    
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card mb-4">
                    <div class="card-header">
                        <i class="bi bi-input-cursor-text me-2"></i>Enter Your Text
                    </div>
                    <div class="card-body">
                        <form id="text-form">
                            <div class="mb-3">
                                <label for="text" class="form-label">Input Text</label>
                                <textarea class="form-control" id="text" rows="8" required 
                                    placeholder="Paste your text here..."></textarea>
                            </div>
                            
                            <div class="row g-3 mb-4">
                                <div class="col-md-4">
                                    <label class="form-label"><i class="bi bi-gear me-2"></i>Mode</label>
                                    <select id="mode" class="form-select">
                                        <option value="extractive">Extractive</option>
                                        <option value="abstractive">Abstractive</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label"><i class="bi bi-list-ol me-2"></i>Sentences</label>
                                    <input type="number" id="num_sentences" class="form-control" value="3" min="1" max="10" />
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label"><i class="bi bi-arrow-down me-2"></i>Min Length</label>
                                    <input type="number" id="min_length" class="form-control" value="30" min="10" />
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label"><i class="bi bi-arrow-up me-2"></i>Max Length</label>
                                    <input type="number" id="max_length" class="form-control" value="130" min="20" />
                                </div>
                            </div>
                            
                            <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                                <button type="submit" class="btn btn-primary px-4">
                                    <i class="bi bi-magic me-2"></i>Generate Summary
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span><i class="bi bi-file-text me-2"></i>Summary</span>
                        <span id="summary-status" class="badge bg-secondary">Ready</span>
                    </div>
                    <div class="card-body">
                        <pre id="summary" class="p-3" style="white-space: pre-wrap; min-height: 150px;">Your summary will appear here...</pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const form = document.getElementById("text-form");
        const summaryElement = document.getElementById("summary");
        const statusElement = document.getElementById("summary-status");
        const modeSelect = document.getElementById("mode");
        const numSentencesInput = document.getElementById("num_sentences");
        
        // Toggle visibility of num_sentences based on mode
        function updateInputsVisibility() {
            if (modeSelect.value === 'extractive') {
                numSentencesInput.closest('.col-md-4').style.display = 'block';
            } else {
                numSentencesInput.closest('.col-md-4').style.display = 'none';
            }
        }
        
        modeSelect.addEventListener('change', updateInputsVisibility);
        updateInputsVisibility(); // Initial call

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            const text = document.getElementById("text").value.trim();
            if (!text) return;
            
            const mode = modeSelect.value;
            const num_sentences = parseInt(document.getElementById("num_sentences").value || "3", 10);
            const min_length = parseInt(document.getElementById("min_length").value || "30", 10);
            const max_length = parseInt(document.getElementById("max_length").value || "130", 10);

            // Update UI
            statusElement.textContent = "Processing...";
            statusElement.className = "badge bg-info";
            summaryElement.textContent = "Generating summary...";
            
            try {
                const response = await fetch("/summarize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        text, 
                        mode, 
                        num_sentences, 
                        min_length, 
                        max_length 
                    }),
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || "Failed to generate summary");
                }
                
                // Update UI with success
                statusElement.textContent = "Done";
                statusElement.className = "badge bg-success";
                summaryElement.textContent = data.summary;
                
                // Smooth scroll to results
                summaryElement.scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                // Update UI with error
                statusElement.textContent = "Error";
                statusElement.className = "badge bg-danger";
                summaryElement.textContent = `Error: ${error.message}`;
                console.error("Error:", error);
            }
        });
        
        // Example text for testing
        document.getElementById("text").addEventListener("focus", function() {
            if (!this.value) {
                this.value = ""; // Clear any default text on focus
            }
        });
    });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """


