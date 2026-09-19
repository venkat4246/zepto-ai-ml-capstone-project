
from fastapi import FastAPI

from graph import create_graph
from models import AskRequest, SupportResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="AI-powered Zepto policy support assistant"
)

support_graph = create_graph()


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant API is running"
    }


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest):

    state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }

    result = support_graph.invoke(state)

    return SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
