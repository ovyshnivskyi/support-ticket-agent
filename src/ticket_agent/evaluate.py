import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ticket_agent.agent import AgentState, build_agent
from ticket_agent.classifier_client import ClassifierClient
from ticket_agent.config import PROJECT_ROOT, Settings
from ticket_agent.schemas import AgentResult, PredictionRequest

GLOBAL_FORBIDDEN_REPLY_PATTERNS: tuple[str, ...] = (
    # "I have forwarded", "we already escalated".
    r"\b(?:i|we)\s+(?:have\s+)?(?:already\s+)?"
    r"(?:forwarded|escalated|submitted)\b",

    # "I've forwarded", "we’ve submitted".
    r"\b(?:i|we)['’]ve\s+(?:already\s+)?"
    r"(?:forwarded|escalated|submitted)\b",

    # "I'm forwarding", "I’m forwarding", "we are escalating".
    r"\b(?:i(?:['’]m|\s+am)|we(?:['’]re|\s+are))\s+"
    r"(?:forwarding|escalating|submitting)\b",

    # "I will forward", "we’ll escalate".
    r"\b(?:i|we)(?:['’]ll|\s+will)\s+"
    r"(?:forward|escalate|submit)\b",

    # Unsupported promises of a quick response.
    r"\b(?:contact|get\s+back\s+to)\s+you\s+(?:shortly|soon)\b",

    # Unsupported promises of future updates.
    r"\b(?:we|they)(?:['’]ll|\s+will)\s+keep\s+you\s+updated\b",
)

@dataclass(frozen=True)
class EvalCase:
    name: str
    title: str
    description: str
    expected_action: str
    expected_sources: tuple[str, ...]
    forbidden_reply_patterns: tuple[str, ...] = ()


CASES = [
    EvalCase(
        name="duplicate_charge_missing_details",
        title="Charged twice",
        description="I was charged twice for my subscription. Please help.",
        expected_action="ask_clarification",
        expected_sources=("KB-001",),
        forbidden_reply_patterns=(
            r"\blast[\s-]+(?:4|four)[\s-]+digits\b",
        ),
    ),
    EvalCase(
        name="duplicate_charge_with_details",
        title="Duplicate subscription charge",
        description=(
            "Both charges are for the same subscription period. "
            "Transaction TX-101 was on September 10, "
            "and TX-102 was on September 11. Please investigate."
        ),
        expected_action="escalate",
        expected_sources=("KB-001",),
        forbidden_reply_patterns=(
            r"\b(?:i|we)\s+(?:have\s+)?(?:already\s+)?"
            r"(?:forwarded|escalated|submitted)\b",
        ),

    ),
    EvalCase(
        name="password_reset",
        title="Forgot my password",
        description=(
            "I forgot my password. I can still access my registered email. "
            "How can I reset the password?"
        ),
        expected_action="reply",
        expected_sources=("KB-002",),
    ),
    EvalCase(
        name="password_reset_without_email",
        title="Cannot access my account",
        description=(
            "I forgot my password and permanently lost access "
            "to my registered email address."
        ),
        expected_action="escalate",
        expected_sources=("KB-002",),
        forbidden_reply_patterns=(
            r"\bfull name\b",
            r"\bphone number\b",
            r"\border numbers?\b",
        ),
    ),
    EvalCase(
        name="unsupported_shipping_request",
        title="Change shipping address",
        description="Please change the shipping address for order ORD-123.",
        expected_action="escalate",
        expected_sources=(),
    ),
]


def check_result(
    case: EvalCase,
    decision: AgentResult,
    messages: list[ToolMessage],
) -> dict[str, bool]:
    searches = [
        message
        for message in messages
        if message.name == "search_knowledge_base"
        and message.status == "success"
    ]

    retrieved_ids = {
        source_id
        for message in searches
        if isinstance(message.content, str)
        for source_id in re.findall(r"\[(KB-\d+)\]", message.content)
    }

    declared_ids = set(decision.source_ids)
    draft_ids = set(re.findall(r"\[(KB-\d+)\]", decision.draft_reply))

    forbidden_patterns = (
            GLOBAL_FORBIDDEN_REPLY_PATTERNS
            + case.forbidden_reply_patterns
    )

    return {
        "correct_action": decision.action == case.expected_action,
        "search_executed": bool(searches),
        "expected_sources": declared_ids == set(case.expected_sources),
        "sources_were_retrieved": (declared_ids | draft_ids) <= retrieved_ids,
        "nonempty_reply": bool(decision.draft_reply.strip()),
        "no_known_unsupported_claims": not any(
            re.search(
                pattern,
                decision.draft_reply,
                flags=re.IGNORECASE,
            )
            for pattern in forbidden_patterns
        ),
    }


async def main() -> int:
    settings = Settings()
    agent = build_agent(settings)
    run_config: RunnableConfig = {"recursion_limit": 8}

    reports: list[dict[str, object]] = []
    passed = failed = errors = 0

    async with httpx.AsyncClient(
        base_url=settings.classifier_base_url,
        timeout=settings.classifier_timeout_seconds,
    ) as http_client:
        classifier = ClassifierClient(http_client)

        for case in CASES:
            started = perf_counter()
            record: dict[str, object] = {
                "name": case.name,
                "expected_action": case.expected_action,
                "expected_sources": list(case.expected_sources),
            }

            try:
                request = PredictionRequest(
                    title=case.title,
                    description=case.description,
                )
                record["ticket"] = request.model_dump(mode="json")

                prediction = await classifier.predict(request)
                record["classification"] = prediction.model_dump(mode="json")

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

                result = await agent.ainvoke(
                    initial_state,
                    config=run_config,
                )
                decision = AgentResult.model_validate(result["result"])

                tool_messages = [
                    message
                    for message in result["messages"]
                    if isinstance(message, ToolMessage)
                ]
                checks = check_result(case, decision, tool_messages)

                record["decision"] = decision.model_dump(mode="json")
                record["checks"] = checks
                record["tool_results"] = [
                    {
                        "name": message.name,
                        "status": message.status,
                        "content": message.content,
                    }
                    for message in tool_messages
                ]

                if all(checks.values()):
                    passed += 1
                    record["status"] = "PASS"
                    print(f"PASS  {case.name}", flush=True)
                else:
                    failed += 1
                    record["status"] = "FAIL"
                    failures = [name for name, ok in checks.items() if not ok]
                    print(f"FAIL  {case.name}: {failures}", flush=True)

            except Exception as exc:
                errors += 1
                record["status"] = "ERROR"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                print(
                    f"ERROR {case.name}: {type(exc).__name__}",
                    flush=True,
                )

            record["duration_seconds"] = round(perf_counter() - started, 2)
            reports.append(record)

    timestamp = datetime.now(UTC)
    output_dir = PROJECT_ROOT / "artifacts" / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"eval_{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    )

    report = {
        "timestamp": timestamp.isoformat(),
        "model": settings.llm_model,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "cases": reports,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nPASS={passed}, FAIL={failed}, ERROR={errors}")
    print(f"Report: {output_path}")

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))