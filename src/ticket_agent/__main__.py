import asyncio
import json

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from ticket_agent.agent import AgentState, build_agent
from ticket_agent.classifier_client import ClassifierClient
from ticket_agent.config import Settings
from ticket_agent.schemas import AgentResult, PredictionRequest


async def main() -> None:
    settings = Settings()

    request = PredictionRequest(
        title="Account locked",
        description=(
            "I cannot log into my account"
            "after changing my password"
        ),
    )

    async with httpx.AsyncClient(
        base_url=settings.classifier_base_url,
        timeout=settings.classifier_timeout_seconds,
    ) as http_client:
        classifier = ClassifierClient(http_client)
        prediction = await classifier.predict(request)

    print("Classification:")
    print(prediction.model_dump_json(indent=2))

    agent = build_agent(settings)

    payload = {
        "ticket": request.model_dump(mode="json"),
        "classification": prediction.model_dump(mode="json"),
    }

    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=False)
            )
        ]
    }

    run_config: RunnableConfig = {
        "recursion_limit": 8,
    }

    try:
        result = await agent.ainvoke(
            initial_state,
            config=run_config,
        )
    except GraphRecursionError:
        print("Agent stopped: graph step limit reached. No draft produced.")
        return

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"\nTool call: {call['name']} {call['args']}")
        elif isinstance(message, ToolMessage):
            print(f"\nTool result:\n{message.content}")

    decision = AgentResult.model_validate(result["result"])

    print("\nAgent result:")
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())