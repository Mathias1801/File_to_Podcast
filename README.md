# PDF-to-Podcast Pipeline

An end-to-end generative AI pipeline that transforms any academic PDF into a fully produced, two-host podcast — complete with structured section extraction, LLM-scripted dialogue, text-to-speech synthesis, and an interactive chat interface for follow-up questions.

---

## What This Project Demonstrates

- Multi-stage LLM prompt engineering using Meta Llama 3.1/3.2 and the xAI Grok API
- GPU-accelerated model inference with Hugging Face Transformers and Accelerate
- Text-to-speech audio generation using Suno Bark with distinct voice presets per speaker
- PDF text extraction, chunking, and structured content parsing
- Knowledge graph construction from unstructured text using NLP (NER, keyword extraction, sentence transformers)
- Graph analysis and visualisation with NetworkX and PyVis
- Function-calling LLM chat interface for interactive document Q&A
- Structured JSON output parsing and serialisation via pickle for cross-notebook data passing

---

## Overview

The pipeline takes a PDF — intended for academic papers — and produces a listenable, engaging two-host podcast covering its introduction, methodology, theory, results, and conclusion. A separate module builds a knowledge graph from the same document and visualises concept relationships. An additional chat notebook allows interactive Q&A against the PDF using Grok's function-calling API.

All notebooks are designed to run in Google Colab with GPU acceleration.

---

## Pipeline

```
target.pdf
    │
    ▼
Step 1 — PDF Preprocessing      Extract and chunk raw text (Llama 3.2-1B)
    │
    ▼
Step 2 — Section Extraction     Restructure into Introduction / Methodology /
    │                           Theory / Results / Conclusion (Llama 3.1-8B)
    ▼
Step 3 — Script Generation      Generate two-host podcast dialogue per section
    │                           as structured list of tuples (Llama 3.1-8B)
    ▼
Step 4 — Text-to-Speech         Synthesise audio using Suno Bark;
                                Speaker 1 = en_speaker_9, Speaker 2 = en_speaker_6
                                Output: per-section .wav files
```

Alongside the main pipeline, two additional modules operate on the same input:

- **Knowledge Graph + EDA** — extracts keywords, builds a weighted concept graph, runs NER via Hugging Face pipelines, and renders an interactive PyVis visualisation
- **Chat Function** — a Grok-powered PDF assistant with a `search_pdf_text` function-calling tool for interactive Q&A

---

## Project Structure

```
├── Step-1-PDF-Preprocess.ipynb   # PDF extraction and chunking
├── Step-2-Section.ipynb          # LLM-based section extraction
├── Step-3-Script.ipynb           # Podcast script generation
├── Step-4-TTS.ipynb              # Bark text-to-speech synthesis
├── Question_Generation           # Multiple-choice quiz generator from sections
├── Knowledge_graph___EDA.ipynb   # Keyword extraction, NER, knowledge graph
├── Chat-function.ipynb           # Grok PDF chat assistant with function calling
└── target.pdf                    # Input document (bring your own)
```

---

## Getting Started

**Prerequisites:** Google Colab with GPU runtime (L4 or better recommended), a Hugging Face account with access to Meta Llama 3.1/3.2, and an xAI API key for the chat module.

### 1. Upload your PDF

Place your target document in the working directory and name it `target.pdf`, or update the `pdf_path` variable in Step 1.

### 2. Run the pipeline in order

```
Step-1-PDF-Preprocess.ipynb   →  produces output text
Step-2-Section.ipynb          →  produces output.txt
Step-3-Script.ipynb           →  produces section0.pkl ... section4.pkl
Step-4-TTS.ipynb              →  produces section audio .wav files
```

Each step reads the output of the previous one. Before running Step 4, rename each section `.pkl` file to `data.pkl` and run the TTS notebook once per section.

### 3. Optional modules

Run `Knowledge_graph___EDA.ipynb` independently on any PDF to produce keyword and concept graph outputs. Run `Chat-function.ipynb` with your xAI API key stored as a Colab secret (`x_ai_API_KEY`) for interactive Q&A.

---

## Configuration

| Notebook | Key Variable | Default | Description |
|---|---|---|---|
| Step 1 | `pdf_path` | `"target.pdf"` | Path to input PDF |
| Step 1 | `max_chars` | `100000` | Character limit for extraction |
| Step 1 | `DEFAULT_MODEL` | `Llama-3.2-1B-Instruct` | Cheap model for preprocessing |
| Step 2–3 | `MODEL` | `Llama-3.1-8B-Instruct` | Main generation model |
| Step 4 | `voice_preset` (S1) | `v2/en_speaker_9` | Bark voice for Speaker 1 |
| Step 4 | `voice_preset` (S2) | `v2/en_speaker_6` | Bark voice for Speaker 2 |
| Knowledge Graph | `top_k` | `20` | Top keywords to extract |

---

## Outputs

**Per-section audio** — one `.wav` file per podcast section (Introduction, Methodology, Theory, Results, Conclusion), stitched from per-line Bark generations.

**Interactive knowledge graph** — a `knowledge_graph.html` file viewable in any browser, with hoverable nodes showing keyword frequency and relationship weights.

**Multiple-choice quiz** — a `questions.pkl` dictionary with 4 questions per section, including answer options and feedback, structured for downstream use in a quiz UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| PDF extraction | PyPDF2 |
| Text generation | Meta Llama 3.1-8B / 3.2-1B via Hugging Face Transformers |
| Chat / function calling | xAI Grok API (OpenAI-compatible) |
| Text-to-speech | Suno Bark (`suno/bark`) |
| Audio processing | pydub, scipy, numpy |
| Knowledge graph | NetworkX, PyVis, sentence-transformers |
| NER | Hugging Face `transformers.pipeline` |
| Accelerated inference | Accelerate, bfloat16, CUDA |
| Runtime | Google Colab (GPU) |

---

## Notes and Limitations

- Step 4 (TTS) processes one section at a time due to Bark's memory footprint — rename section `.pkl` files to `data.pkl` before each run.
- The pipeline uses Llama 3.1-8B throughout. Upgrading to 3.1-70B or 3.1-405B would improve script quality but requires significantly more GPU memory.
- Bark's `en_speaker_6` is the highest-quality English voice; `en_speaker_9` is the only available female English voice in the v2 preset set.
- The Question Generation notebook outputs structured JSON — the format is designed to plug directly into a multiple-choice quiz front-end.
