"""Safe tool-execution canaries for end-to-end prompt-injection evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .tool_policy import ToolPolicy


@dataclass
class CanaryEvent:
    requested_tool: str
    arguments: dict[str, Any]
    policy_action: str
    rule_id: str
    executed: bool


class CanaryToolExecutor:
    """Records attempted calls; never performs real external side effects."""

    def __init__(self, policy: ToolPolicy):
        self.policy = policy
        self.events: list[CanaryEvent] = []

    def execute(self, tool: str, arguments: dict[str, Any], confirmed: bool = False) -> dict:
        decision = self.policy.evaluate(tool, arguments)
        executed = decision.action == "allow" or (
            decision.action == "require_confirmation" and confirmed
        )
        event = CanaryEvent(tool, arguments, decision.action, decision.rule_id, executed)
        self.events.append(event)
        return {"canary": True, "executed": executed, "decision": asdict(decision)}

    @property
    def unauthorized_execution_count(self) -> int:
        return sum(event.executed and event.policy_action != "allow" for event in self.events)
