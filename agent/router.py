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

IMPORTANT: If a previous agent turn is provided, short follow-up queries
("show me more", "show me 3 more", "what about X?", "and refunds?", "how many?")
must be treated as continuations of the conversation and classified as structured.
Only classify as out_of_scope if the query is clearly unrelated to customer-service data analysis.

Examples:
"What categories exist in the dataset?"                 → structured
"How many refund requests did we get?"                  → structured
"Show me 3 examples from the SHIPPING intent"           → structured
"Show me 3 more"                                        → structured
"Show me more"                                          → structured
"What about refunds?"                                   → structured
"And the ACCOUNT category?"                             → structured
"What is the intent distribution in ACCOUNT?"           → structured
"What do you remember about me?"                        → structured
"What do you know about me?"                            → structured
"Summarize the FEEDBACK category"                       → unstructured
"How do agents typically handle cancellation requests?" → unstructured
"Who won the 2024 Champions League?"                    → out_of_scope
"Write me a poem about customer service"                → out_of_scope
"What's the best CRM software for complaints?"          → out_of_scope
"""),
    ("human", "Previous agent turn (if any): {context}\n\nUser query: {query}"),
])


def classify_query(query: str, context: str = "") -> QueryType:
    """Classify a user query as structured, unstructured, or out_of_scope.

    Args:
        query:   The user's latest message.
        context: Optional last agent response — helps classify short follow-ups correctly.

    Tries structured output first; falls back to raw-text parsing if the model
    does not support function calling or returns an unexpected format.
    """
    try:
        chain = _PROMPT | get_router_llm().with_structured_output(RouteDecision)
        result = chain.invoke({"query": query, "context": context or "(none)"})
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
