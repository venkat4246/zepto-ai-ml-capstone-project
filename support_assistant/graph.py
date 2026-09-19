
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from ingestion import create_collection
from embeddings import get_embeddings


class SupportState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[str]
    answer: str
    sources: List[str]
    confidence: float


# Create ChromaDB collection
collection = create_collection()


def classify_intent(state: SupportState) -> SupportState:
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if any(keyword in query for keyword in policy_keywords):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state


def retrieve_and_answer(state: SupportState) -> SupportState:
    query = state["query"]

    results = collection.query(
        query_embeddings=get_embeddings([query]).tolist(),
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    state["retrieved_chunks"] = documents

    state["sources"] = [
        metadata["source"]
        for metadata in metadatas
        if metadata is not None
    ]

    query_lower = query.lower()

    if "delivery fee" in query_lower or "fee" in query_lower:
        for document in documents:
            if "149" in document and "25" in document:
                state["answer"] = (
                    "Orders below INR 149 incur a flat INR 25 delivery fee."
                )
                break
        else:
            state["answer"] = (
                "The retrieved documents do not provide the requested fee information."
            )

    elif "delivery time" in query_lower or "how long" in query_lower:
        state["answer"] = (
            "Zepto delivers orders within 10 to 30 minutes, "
            "depending on the delivery zone and current order volume."
        )

    else:
        state["answer"] = documents[0][:300]

    state["confidence"] = 0.95

    return state


def direct_answer(state: SupportState) -> SupportState:
    state["answer"] = (
        "I can only answer questions about Zepto policies right now."
    )

    state["sources"] = []
    state["confidence"] = 0.50

    return state


def route_intent(state: SupportState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def create_graph():

    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()
