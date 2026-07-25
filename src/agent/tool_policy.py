"""Deterministic authorization boundary for agent tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

PolicyAction = Literal["allow", "deny", "require_confirmation"]

_SECRET_NAMES = re.compile(
    r"(api[_-]?key|token|secret|password|credential|private[_-]?key|\.env|ssh)",
    re.IGNORECASE,
)
_DESTRUCTIVE_SHELL = re.compile(r"\b(rm|rmdir|unlink|shred|format|mkfs|git\s+reset)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ToolDecision:
    action: PolicyAction
    reason: str
    rule_id: str


@dataclass
class ToolPolicy:
    """Policy enforced in code, independently of model instructions."""

    allowed_tools: set[str] = field(default_factory=lambda: {"search", "read_url"})
    allowed_domains: set[str] = field(default_factory=set)
    confirmation_tools: set[str] = field(
        default_factory=lambda: {"email_send", "calendar_write", "file_write", "shell"}
    )

    def evaluate(self, tool: str, arguments: dict[str, Any]) -> ToolDecision:
        if tool not in self.allowed_tools and tool not in self.confirmation_tools:
            return ToolDecision("deny", f"Tool '{tool}' is not allowlisted.", "tool_not_allowlisted")

        serialized = repr(arguments)
        if _SECRET_NAMES.search(serialized):
            return ToolDecision("deny", "Arguments reference secret-bearing material.", "secret_reference")

        if tool == "shell":
            command = str(arguments.get("command", ""))
            if _DESTRUCTIVE_SHELL.search(command):
                return ToolDecision("deny", "Destructive shell command blocked.", "destructive_shell")
            return ToolDecision("require_confirmation", "Shell execution requires user confirmation.", "confirm_shell")

        url = arguments.get("url")
        if url:
            parsed = urlparse(str(url))
            if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                return ToolDecision("deny", "Malformed or unsupported URL.", "invalid_url")
            if self.allowed_domains and parsed.hostname not in self.allowed_domains:
                return ToolDecision("deny", "Domain is outside the configured allowlist.", "domain_not_allowlisted")

        if tool in self.confirmation_tools:
            return ToolDecision(
                "require_confirmation", f"Tool '{tool}' changes external state.", "confirm_side_effect"
            )
        return ToolDecision("allow", "Tool call satisfies deterministic policy.", "allow")


def redact_secrets(value: str) -> str:
    """Redact common credential assignments before logs or model context."""
    patterns = (
        r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"sk-[A-Za-z0-9_-]{16,}",
    )
    redacted = value
    for pattern in patterns:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted
