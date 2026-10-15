# AI Study Assistant (RAG + Grok)

Ask questions from your **5 PDF chapters**. The app creates **embeddings**, **saves them locally**, **retrieves** the
most relevant passages and uses **Grok (xAI)** to write an accurate answer with sources.

## How it works (RAG pipeline)
```
 PDF chapters ──► pdf_loader.py ──► chunks (about 900 characters, 150 overlap, with chapter + page)
                                         │
                                         ▼
                       embedder.py  (sentence-transformers all-MiniLM-L6-v2, runs locally, 384 numbers per chunk)
                                         │
                                         ▼
                  vector_store.py  ──► saved on disk: vector_store/embeddings.npy + chunks.json + index_meta.json
                                         │
 Question ──► embedding ──► cosine similarity ──► top-4 chunks  (rag.py: retrieve)
                                         │
                                         ▼
          prompt = rules + numbered context + question ──► Grok API (grok_client.py) ──► answer with [1] [2] sources
```
| Requirement | Where |
|---|---|
| Generate embeddings | `embedder.py` |
| Save them locally | `vector_store.py` (`vector_store/` folder; reused on the next run, rebuilt automatically when PDFs change) |
| Retrieve relevant information | `VectorStore.search()` and `StudyAssistant.retrieve()` |
| Use Grok for accurate answers | `grok_client.py` + the prompt in `rag.py` (answer only from context, cite sources, say "not found" otherwise) |

Extras: follow-up questions (chat memory), `/summary` and `/quiz` study tools, Streamlit web app, automatic tests.

## Setup
1. Install Python 3.10 or newer.
2. In the project folder:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Get an API key from https://console.x.ai (your xAI account may need credits) and create a file named `.env`
   (copy `.env.example`):
   ```
   XAI_API_KEY=xai-your-key-here
   ```
4. **Add your 5 chapters:** delete the 5 sample PDFs in `chapters/` and copy your own PDF files there.
   (The samples are about "Introduction to AI" and are only for trying the project.)

## Run
| What | Command |
|---|---|
| Chat in the terminal | `python study_assistant.py` |
| One question | `python study_assistant.py --ask "What is overfitting?"` |
| Web app | `streamlit run app.py` |
| Force new embeddings | `python study_assistant.py --rebuild` |
| Run the tests (no internet / key needed) | `python tests.py` |

The first run downloads the embedding model (about 90 MB) and creates the embeddings; later runs load them from disk.
In the chat you can also type `/summary <topic>`, `/quiz <topic>` and `/sources`.

## Settings (config.py / .env)
- `GROK_MODEL` - model name (default `grok-4`). If you get a "model not found" error, use a model name from https://docs.x.ai.
- `MIN_SCORE` - chunks less similar than this are ignored (default 0.10).
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` - chunking and retrieval settings.

## Troubleshooting
- **"No API key found"**: check that `.env` exists in the project folder and contains `XAI_API_KEY=...`.
- **401 / 403 / 404 / 429 errors**: wrong key, no access to the model, wrong model name, or no credits. The message says which one.
- **"has no selectable text"**: the PDF is a scan (pictures). Run OCR on it first, then add it again.
- **Wrong or odd answers**: increase `TOP_K` to 5-6 or reduce `CHUNK_SIZE`, then run with `--rebuild`.

## Possible viva questions
- **What is RAG and why use it?** It retrieves relevant text first and gives it to the LLM, so answers come from your chapters, hallucination is reduced, and no retraining is needed.
- **What is an embedding?** A vector of numbers that represents the meaning of a text. Similar meanings are close together.
- **Why chunk the documents?** An embedding of a whole chapter mixes many topics; small chunks give precise matches and fit in the prompt.
- **Why overlap?** So a sentence or idea at the edge of a chunk is not lost.
- **How is similarity computed?** Cosine similarity. Vectors are normalised, so a dot product gives it.
- **Why save the embeddings?** Creating them is the slow step; saved embeddings make start-up fast, and a fingerprint of the PDFs triggers a rebuild only when needed.
- **How do you reduce hallucination?** The prompt allows only the supplied context, asks for citations, and a "not found" answer is required when the context lacks the answer.
