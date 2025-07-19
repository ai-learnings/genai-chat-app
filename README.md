# GenAI Chat App

This is a GenAI chat application that uses:

- **Locally deployed LLM**: Ollama + Llama3.1
- **Optional Gemini-2.0-Flash** integration
- **RAG (Retrieval Augmented Generation)** with MongoDB and embeddings using `mxbai-embed-large`
- **Web search/crawling** to redefine context for user queries

## Features

- Chat with LLMs locally or via Gemini
- Store and search embeddings in MongoDB
- Context enrichment via web crawling

## Getting Started

1. Deploy Ollama and MongoDB locally
2. Install dependencies in your Python environment
3. Configure environment variables
4. Run `main.py` and follow the prompts

## Environment Variables

Create a `.env` file in the root directory with the following configuration:

```properties
# MongoDB Configuration
MONGODB_URI="mongodb://localhost:27017/?directConnection=true"
MONGODB_DATABASE="llm-vec-embeding-db"
MONGODB_COLLECTION="embeddings"

# Ollama Configuration
OLLAMA_BASE_URL="localhost:11434"
OLLAMA_MODEL="llama3.1:8b"
OLLAMA_EMBEDDING_MODEL="mxbai-embed-large"
OLLAMA_TEMPERATURE=0.8
OLLAMA_NUM_PREDICT=256

# Optional API Keys (if using these services)
# VOYAGE_API_KEY=""
# GEMINI_API_KEY=""
```

> Note: Make sure to add `.env` to your `.gitignore` file to keep sensitive information secure.

## MongoDB indexes
Search vector index: llm-vec-embeding
```
{
  "fields": [
    {
      "type": "vector",
      "path": "data_embeded",
      "numDimensions": 1024,
      "similarity": "cosine"
    }
  ]
}
```

## License

MIT