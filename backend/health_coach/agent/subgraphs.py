"""Composable LangGraph subgraphs for the health coach."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph

from ..core.health_normalizer import normalize_health_result
from ..integrations.nutrition import compose_nutrition_reply, search_food_nutrition
from .actions import QUERY_INTENTS, execute_health_action
from .engine import AIEngine, Intent


class ResearchState(dict):
    pass


def build_research_subgraph(*, engine: AIEngine, research_fn: Callable[..., str]):
    """Tavily research → LLM answer."""

    def research_answer(state: dict[str, Any]) -> dict[str, Any]:
        return {"final_reply": research_fn(state)}

    graph = StateGraph(dict)
    graph.add_node("research_answer", research_answer)
    graph.set_entry_point("research_answer")
    graph.add_edge("research_answer", END)
    return graph.compile()


def build_health_query_subgraph(*, engine: AIEngine, client, llm_context_fn):
    """Google Health query → normalize → summarize."""

    def finalize_query(state: dict[str, Any]) -> dict[str, Any]:
        ctx = llm_context_fn(state)
        reply = state.get("conversational_reply", "")
        api_result = state.get("api_result")
        intent_name = state.get("intent", "")
        try:
            intent = Intent(intent_name)
        except ValueError:
            return {"final_reply": reply.strip()}
        if intent in QUERY_INTENTS and api_result is not None:
            data_type = state.get("payload", {}).get("data_type")
            normalized = (
                normalize_health_result(data_type, api_result) if data_type else api_result
            )
            reply = engine.summarize_health_data(
                user_text=state["user_text"],
                draft_reply=reply,
                api_result=normalized,
                conversation_context=ctx["conversation_context"],
                user_profile_context=ctx["user_profile_context"],
            )
        return {"final_reply": reply.strip()}

    graph = StateGraph(dict)
    graph.add_node("finalize_query", finalize_query)
    graph.set_entry_point("finalize_query")
    graph.add_edge("finalize_query", END)
    return graph.compile()


def build_local_coach_subgraph(*, client, execute_local_fn):
    """SQLite-only coach actions (plans, mood, goals, undo)."""

    def run_local(state: dict[str, Any]) -> dict[str, Any]:
        result = execute_local_fn(
            state["intent"],
            state.get("payload", {}),
            client=client,
            sender_phone=state.get("sender_phone"),
            user_text=state.get("user_text", ""),
        )
        reply = result.get("message", "") if result else ""
        return {"api_result": result, "final_reply": reply}

    graph = StateGraph(dict)
    graph.add_node("run_local", run_local)
    graph.set_entry_point("run_local")
    graph.add_edge("run_local", END)
    return graph.compile()
