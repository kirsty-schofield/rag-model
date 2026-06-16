# Local RAG Pipeline with LangChain and Qwen

This repository contains a self-contained, privacy-focused Retrieval-Augmented Generation (RAG) pipeline. It uses LangChain for data ingestion, Chroma DB for local vector storage, and the Qwen 2.5 (0.5B) model to answer questions entirely on your local machine, without the need for API keys. 

## Features

- Runs completely offline using open-source models via Hugging Face Transformers.
- Evaluates vector distance scores to filter out irrelevant context before sending prompts to the LLM.
- Formats context to track and inject document page numbers into the prompt.
- Guardrails stop the chain and trigger a fallback message if no relevant document chunks pass the quality threshold.
