from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from agent.llm import get_router_llm


class QueryType(str, Enum):
    structured   = "structured"
    unstructured = "unstructured"
    out_of_scope = "out_of_scope"


class RouteDecision(BaseModel):
    """Structured output schema for the query router."""
    query_type: QueryType


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You classify user queries for a customer-service dataset analysis agent.

The agent can ONLY answer questions about the Bitext Customer Service dataset —
a collection of customer support queries and agent responses, grouped by category and intent.

Classify as exactly one of:
- structured:   questions with concrete, data-driven answers (counts, lists, examples, distributions)
- unstructured: open-ended questions that need summarization or synthesis of dataset content
- out_of_scope: anything unrelated to the customer service dataset

Examples:
"What categories exist in the dataset?"              → structured
"How many refund requests did we get?"               → structured
"Show me 3 examples from the SHIPPING intent"        → structured
"What is the intent distribution in ACCOUNT?"        → structured
"Summarize the FEEDBACK category"                    → unstructured
"How do agents typically handle cancellation requests?" → unstructured
"Who won the 2024 Champions League?"                 → out_of_scope
"Write me a poem about customer service"             → out_of_scope
"What's the best CRM software for complaints?"       → out_of_scope
"""),
    ("human", "{query}"),
])


def classify_query(query: str) -> QueryType:
    """Classify a user query as structured, unstructured, or out_of_scope.

    Tries structured output first; falls back to raw-text parsing if the model
    does not support function calling or returns an unexpected format.
    """
    try:
        chain = _PROMPT | get_router_llm().with_structured_output(RouteDecision)
        result = chain.invoke({"query": query})
        return QueryType(result.query_type)
    except Exception:
        raw = get_router_llm().invoke(
            "Reply with exactly one word — structured, unstructured, or out_of_scope — "
            f"to classify this query about a customer-service dataset:\n\n{query}"
        ).content.strip().lower()
        for qt in QueryType:
            if qt.value in raw:
                return qt
        return QueryType.out_of_scope
