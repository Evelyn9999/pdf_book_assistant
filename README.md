# PDF Book Assistant

## What this project does

PDF Book Assistant is a desktop AI application for asking questions about a loaded textbook PDF.

It is designed for study use cases such as:
- definition questions (e.g., "What is TCP?")
- location questions (e.g., "Which chapter explains transport layer?")
- structure questions (e.g., "How many chapters are in the book?")
- overview questions (e.g., "What is this book about?")

The app shows:
- a short answer
- relevant passages with page/chapter/section context
- routing/debug status (`channel` and `debug reason`)

## Main idea (high level)

The system uses a dual-channel approach:
- **Structured channel**: heading/chapter/section-based answering for count/location/overview type questions
- **Semantic channel**: hybrid retrieval (BM25 + embeddings) + reranker + extractive answer selection

It also includes:
- confidence gate (avoid overconfident wrong answers)
- domain gate (reject out-of-book/general-chat questions safely)

## Requirements

- Python 3.10+ (tested with Python 3.13 on Windows)
- Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## Run the application

From `pdf_book_assistant` directory:

```bash
py app.py
```

Then:
1. Click **Select PDF**
2. Click **Load PDF**
3. Enter a question and click **Ask**

## Baseline demo (TF-IDF alternative)

Run:

```bash
py baseline_tfidf_demo.py --pdf "data/Computer Networking .pdf" --query "What is TCP?"
```

Note: the sample filename above includes a space before `.pdf`. Adjust if your file name differs.

## Generate comparison chart

Create/edit CSV:

```bash
py compare_plot.py
```

Then generate chart:

```bash
py compare_plot.py --csv evaluation_summary.csv --out comparison_chart.png
```

## Project structure

- `app.py` - GUI and main workflow
- `answerer.py` - answer extraction, routing support, confidence/domain gating
- `retriever_tfidf.py` - hybrid retriever + reranker
- `structured_channel.py` - structure-first QA channel
- `structure_index.py` - strict chapter heading extraction
- `chunker.py` - paragraph-aware chunking with overlap
- `text_cleaner.py` - PDF text cleanup
- `baseline_tfidf_demo.py` - TF-IDF baseline script
- `compare_plot.py` - performance chart script

