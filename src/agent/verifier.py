"""
Research Verifier

Performs claim-level grounding: each factual claim in the synthesized answer
is checked against the retrieved evidence. Claims that cannot be traced to a
specific extract are flagged as unverified.

This is the agent's process-level reward signal. Rather than scoring only the
final answer, we score every individual claim. The result is a hallucination
rate metric that is more granular and actionable than a binary pass/fail.

Two failure modes this catches:
  1. Fabricated facts — claims with no basis in any retrieved evidence
  2. Citation drift — claims that cite a source but the cited source does
     not actually support the claim (a subtler form of hallucination)
"""

import json
import re
from anthropic import Anthropic

from .models import Evidence, Claim


VERIFY_PROMPT = """You are a fact-checker verifying claims against source evidence.

For each claim below, determine whether it is supported by the provided evidence.
A claim is VERIFIED if the evidence contains text that directly supports it.
A claim is UNVERIFIED if no evidence supports it, or if it goes beyond what the evidence states.

Evidence:
{evidence_block}

Claims to verify:
{claims_block}

Respond with a JSON array. Each element corresponds to one claim (in the same order):
[
  {{
    "claim": "exact claim text",
    "verified": true,
    "supporting_excerpt": "the specific sentence from evidence that supports it, or null if unverified"
  }},
  ...
]"""


EXTRACT_CLAIMS_PROMPT = """Extract all distinct factual claims from this research answer.
A factual claim is a specific assertion about the world that could be verified or falsified.
Do not include meta-statements like "research shows" or "according to sources".

Answer text:
{answer}

Respond with a JSON array of claim strings:
["claim 1", "claim 2", ...]

Extract between 3 and 15 claims. Focus on the most specific factual assertions."""


class ResearchVerifier:
    def __init__(self, client: Anthropic, model: str):
        self.client = client
        self.model = model

    def verify(
        self,
        answer_text: str,
        evidence: list[Evidence],
        sources: list[dict],
    ) -> tuple[list[Claim], list[str]]:
        """
        Verify each factual claim in the answer against the retrieved evidence.

        Returns:
            (claims, unverified_claim_texts) tuple
        """
        # Step 1: Extract claims from the answer
        claims_raw = self._extract_claims(answer_text)
        if not claims_raw:
            return [], []

        # Step 2: Build evidence block for verification
        good_evidence = [e for e in evidence if e.search_successful and e.extracted_text]
        evidence_lines = []
        for i, ev in enumerate(good_evidence):
            evidence_lines.append(f"[E{i+1}] {ev.title}: {ev.extracted_text}")
        evidence_block = "\n".join(evidence_lines)

        # Step 3: Verify all claims in a single pass
        claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims_raw))

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": VERIFY_PROMPT.format(
                        evidence_block=evidence_block,
                        claims_block=claims_block,
                    ),
                }
            ],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        verified_results = json.loads(raw)

        claims = []
        unverified_texts = []

        for item in verified_results:
            # Extract citation numbers from claim text (e.g., [1], [2])
            citation_matches = re.findall(r"\[(\d+)\]", item.get("claim", ""))
            citation_numbers = [int(m) for m in citation_matches]

            claim = Claim(
                text=item["claim"],
                citation_numbers=citation_numbers,
                verified=item.get("verified", False),
                supporting_excerpt=item.get("supporting_excerpt"),
            )
            claims.append(claim)

            if not claim.verified:
                unverified_texts.append(claim.text)

        return claims, unverified_texts

    def _extract_claims(self, answer_text: str) -> list[str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_CLAIMS_PROMPT.format(answer=answer_text),
                }
            ],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)
