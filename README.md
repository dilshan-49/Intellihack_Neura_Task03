# 🧠 LLM Fine-Tuning Challenge – Qwen 2.5-3B for AI Research QA

This project showcases the end-to-end pipeline of fine-tuning the Qwen 2.5-3B model for AI research question-answering using synthetic data generation, vector retrieval, and model training — all within resource-constrained environments.

---

## 📌 Highlights

- 🔍 **Document Preprocessing**: Extracted and chunked PDF/Markdown documents using LangChain.
- 🧠 **Embeddings**: Generated vector representations using `multi-qa-mpnet-base-dot-v1`.
- 📁 **FAISS Retrieval**: Enabled semantic chunk retrieval for building QA context.
- 🤖 **Synthetic QA Generation**: Used prompt engineering and Qwen model to generate and evolve QA pairs.
- 📊 **Training**: Fine-tuned Qwen 2.5-3B using Hugging Face’s TRL + Unsloth for optimized performance in Google Colab.
- 💬 **RAG Inference**: Combined FAISS similarity search with trained model for context-aware response generation.
- 🧠 **LSTM Integration (WIP)**: Experimented with LSTM memory to extend chatbot capabilities.

---

## 🧪 Tech Stack

| Component         | Technology                      |
|------------------|----------------------------------|
| Language Model    | Qwen2.5-3B + Unsloth             |
| Embeddings        | Sentence-Transformers (`mpnet`) |
| Retriever         | FAISS                           |
| Dataset Handling  | LangChain, Markdown, PDFs       |
| Training          | Hugging Face TRL, PyTorch       |
| Hosting           | Google Colab                    |

---

## 🧠 Results

- Trained model can answer synthetic and real AI research questions with relevant context.
- Generated and saved over 800 QA samples in JSON format.
- Achieved a working RAG setup with fast retrieval + compact generation via 4-bit quantization.

---

## 📁 Repo & Resources

- 🤗 Model: [huggingface.co/Shehan909/Neura](https://huggingface.co/Shehan909/Neura)
- 🔗 References:
  - [Synthetic Data via LLMs – Confident AI](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms)
  - [Unsloth Qwen2.5 Fine-tuning Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb)

---

## 📌 Future Work

- Integrate persistent chat memory (LSTM or FAISS hybrid).
- Improve query evolution diversity.
- Explore model distillation or smaller architectures for deployment.

---



