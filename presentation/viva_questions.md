# Viva Questions: College Knowledge Assistant (RAG)

Short answers based on our actual implementation (`rag_pipeline.py`, `app.py`, `.env`).

---

**1. Why did you use RAG instead of fine-tuning the LLM on college documents?**
Fine-tuning needs training data, GPU time and a retrain whenever a document changes. RAG keeps the model unchanged and gives it the relevant text at question time. A new PDF works immediately, answers can be traced to source pages, and the system can say "not available" when the document has no answer.

**2. What happens when a PDF is uploaded? (ingestion pipeline)**
`process_pdf()` checks that Ollama is running and both models are pulled. Then `load_pdf()` uses pypdf `PdfReader` to extract each page's text (skipping empty pages), and `split_documents()` makes 500-char chunks with 100-char overlap. Finally `FaissVectorStore` embeds the chunks with nomic-embed-text, L2-normalises them and adds them to a FAISS `IndexFlatIP`. For `sample.pdf`: 12 text pages became 17 chunks.

**3. Why chunk size 500 and overlap 100?**
Small chunks keep each vector focused on one idea, which makes retrieval more precise. 4 chunks × 500 characters also fit easily in the LLM's 4,096-token context. The 100-character overlap means a sentence cut at a boundary still appears whole in one of the two chunks. Both values are configurable in `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`).

**4. Why `RecursiveCharacterTextSplitter`?**
It tries separators in order: `"\n\n"` (paragraph), `"\n"` (line), `" "` (word), then `""` (character). Chunks therefore break at natural boundaries where possible instead of mid-word. Each chunk keeps its page number in `metadata`, which is how we show sources.

**5. Why do you add `search_document:` and `search_query:` prefixes?**
nomic-embed-text was trained with task prefixes. Documents get `search_document:` and questions get `search_query:`, so the model knows which side of the search each text is on. Leaving them out gives less accurate similarity.

**6. What is an embedding, and what is its dimension here?**
An embedding is a fixed-length vector of numbers that represents the meaning of a text. Texts with similar meaning have vectors pointing in similar directions. nomic-embed-text produces 768-dimensional vectors.

**7. What does FAISS do, and does it store the text?**
FAISS stores and searches vectors only, not text. Each vector gets an id 0…n-1 in insertion order. We keep the chunks in a Python list in the same order, so FAISS id `i` maps to `self.chunks[i]`.

**8. Why `IndexFlatIP` with `normalize_L2`, and why is that cosine similarity?**
`normalize_L2` scales every vector to length 1. For unit vectors, cos θ = q·d / (‖q‖‖d‖) = q·d, so the inner product (IP) is the cosine similarity. "Flat" means exact brute-force search: the query is compared with every vector. That is fast enough for thousands of chunks.

**9. When would you replace `IndexFlatIP` with another index?**
When there are hundreds of thousands to millions of vectors, exact search becomes slow. Approximate indexes such as IVF (clustering) or HNSW (graph) trade a little recall for much faster search.

**10. Why Top-K = 4?**
It's a trade-off. Too few chunks may miss the answer; too many add noise and use up the context window, and a small model can get distracted. 4 chunks of about 500 characters gave enough context in our tests, and `TOP_K` is configurable in `.env`.

**11. How do you reduce hallucination?**
- The system prompt allows only facts from the CONTEXT.
- A fixed reply is required when the answer is missing ("This information is not available in the uploaded document.").
- `temperature=0` gives deterministic, factual output.
- The context sits right next to the question in the human message.
- Sources are hidden when the answer is "not available".

In our self-test, both out-of-scope questions (FIFA World Cup, capital of France) were refused. Hallucination is reduced, not eliminated: a 3B model can still make mistakes.

**12. What is LCEL and what does `RAG_PROMPT | llm | StrOutputParser()` do?**
LCEL is the LangChain Expression Language. The `|` operator pipes one step's output into the next. The prompt template fills `{context}`, `{question}` and `{not_found}`, ChatOllama generates a message, and StrOutputParser turns it into a plain string.

**13. How does the app avoid re-processing the PDF on every message?**
Streamlit re-runs the whole script on every interaction. We store the FAISS store, file id and chat history in `st.session_state`. Indexing runs only when `uploaded_file.file_id` differs from the stored one.

**14. What is the difference between the embedding model, FAISS and the LLM?**
The embedding model (nomic-embed-text) only turns text into vectors. FAISS only stores vectors and finds the nearest ones. The LLM (llama3.2:3b) only writes the answer from the retrieved text. None of them does another's job.

**15. What happens with a scanned PDF?**
pypdf extracts no text from image-only pages. If no page has text, `load_pdf()` raises a `ValueError` ("may be a scanned/image-only PDF") and the app shows it as an error. Supporting these files needs OCR, which is listed as future work.

**16. Why run locally with Ollama instead of a cloud API?**
There's no API key or per-request cost, it works offline, and college documents never leave the machine. The trade-offs are that a small local 3B model is less capable than large cloud models, and answers take a few seconds on laptop hardware.

**17. How would you evaluate the system properly?**
Build a labelled set of questions with their correct pages and answers. Measure retrieval with Hit-rate / Recall@K (is the right page in the Top-4?). Measure generation with answer correctness and faithfulness, judged manually or with a framework such as RAGAS. So far we have only a 5-question manual self-test, not a benchmark.
