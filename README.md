# Deploy FastAPI on Render

Use this repo as a template to deploy a Python [FastAPI](https://fastapi.tiangolo.com) service on Render.

See https://render.com/docs/deploy-fastapi or follow the steps below:

## Manual Steps

1. You may use this repository directly or [create your own repository from this template](https://github.com/render-examples/fastapi/generate) if you'd like to customize the code.
2. Create a new Web Service on Render.
3. Specify the URL to your new repository or this repository.
4. Render will automatically detect that you are deploying a Python service and use `pip` to download the dependencies.
5. Specify the following as the Start Command.

    ```shell
    uvicorn main:app --host 0.0.0.0 --port $PORT
    ```

6. Click Create Web Service.

Or simply click:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/render-examples/fastapi)

# GenAI RAG Application on Render

A Python-based FastAPI Generative AI application on Render.

This project integrates:

- **FastAPI** for API development
- **LangChain** for orchestration
- **Google Gemini (Gemini 1.5 Pro)** as the LLM
- **Google Generative AI Embeddings**
- **AstraDB Vector Store** for semantic document retrieval
- **RAG (Retrieval-Augmented Generation)** architecture for contextual question answering

---

# Project Architecture

```text
User Question
      │
      ▼
FastAPI Endpoint
      │
      ▼
LangChain Pipeline
      │
      ├── Generate Embeddings
      │
      ├── Search AstraDB Vector Store
      │
      └── Retrieve Relevant Documents
      │
      ▼
Gemini 1.5 Pro LLM
      │
      ▼
Context-Aware Answer
```

# GenAI Model Explanation

This project uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain.

The workflow is as follows:

- User submits a question.
- The question is converted into embeddings using Google's embedding model.
- AstraDB performs similarity search on stored vectors.
- Relevant documents are retrieved.
- LangChain combines retrieved context with the user query.
- Gemini 1.5 Pro generates a context-aware response.

This ensures:
- More accurate answers
- Reduced hallucinations
- Domain-specific responses
- Context-aware generation
