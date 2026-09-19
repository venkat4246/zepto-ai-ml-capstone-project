
# Zepto Support Assistant

## Overview

The Zepto Support Assistant is an AI-powered customer support system that answers questions using Zepto policy documents.

## Features

- 8 Zepto policy documents
- MiniLM text embeddings
- ChromaDB vector retrieval
- LangGraph workflow
- Intent classification
- Pydantic structured responses
- FastAPI REST API
- Docker support
- Mock LLM approach with no API key required

## Architecture

```text
User Query
    ↓
FastAPI /ask
    ↓
Intent Classification
    ↓
Policy Question?
   / \
 Yes  No
  ↓    ↓
ChromaDB  Direct Answer
  ↓
MiniLM Retrieval
  ↓
Answer + Sources + Confidence
  ↓
Pydantic JSON Response
