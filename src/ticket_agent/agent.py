import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langchain_groq import ChatGroq
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ticket_agent.config import Settings
from ticket_agent.tools import search_knowledge_base
from typing import NotRequired

from ticket_agent.schemas import AgentResult
from ticket_agent.prompts import loader


class AgentState(MessagesState):
    result: NotRequired[AgentResult]

def build_agent(
    settings: Settings,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState,]:
    model = ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=2048,
        timeout=30.0,
        max_retries=2,
    )

    tools = [search_knowledge_base]
    model_with_tools = model.bind_tools(tools, parallel_tool_calls=False,)

    structured_model = model.with_structured_output(
        AgentResult,
        method="json_schema",
        strict=True,
    )

    path = settings.prompt_path
    system_prompt, finalize_prompt = loader.load_prompts_from_file(path)


    async def finalize(state: AgentState) -> dict[str, AgentResult]:
        initial_message = state["messages"][0]

        if not isinstance(initial_message, HumanMessage):
            raise ValueError("Expected the initial ticket message.")

        if not isinstance(initial_message.content, str):
            raise ValueError("Expected a JSON string in the ticket message.")

        original_payload = json.loads(initial_message.content)

        articles = [
            message.content
            for message in state["messages"]
            if isinstance(message, ToolMessage)
            and message.name == "search_knowledge_base"
            and message.status == "success"
            and isinstance(message.content, str)
        ]

        evidence = {
            "ticket": original_payload["ticket"],
            "classification": original_payload["classification"],
            "retrieved_articles": articles,
        }

        response = await structured_model.ainvoke(
            [
                SystemMessage(content=finalize_prompt),
                HumanMessage(
                    content=json.dumps(evidence, ensure_ascii=False)
                ),
            ]
        )

        decision = AgentResult.model_validate(response)
        return {"result": decision}

    async def call_model(state: AgentState) -> MessagesState:
        response = await model_with_tools.ainvoke(
            [
                SystemMessage(content=system_prompt),
                *state["messages"],
            ]
        )

        return {"messages": [response]}

    graph = StateGraph(AgentState)

    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "model")

    graph.add_conditional_edges(
        "model",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "finalize",
        },
    )

    graph.add_edge("tools", "model")
    graph.add_edge("finalize", END)

    return graph.compile()