from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .domain import content_hash, normalize_name, read_json, write_json


CANDIDATE_TYPES = {
    "abbreviation", "alias", "former_name", "subsidiary", "affiliate",
    "regional", "product_service", "informal",
}


class ExpansionService:
    """Cached, optional Azure OpenAI expansion. Never used for pair decisions."""

    def __init__(self, config: dict[str, Any], prompt_path: str | Path, cache_path: str | Path):
        self.config = config
        self.mode = str(config.get("mode", "off"))
        self.prompt_path = Path(prompt_path)
        self.cache_path = Path(cache_path)
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.cache: dict[str, dict[str, Any]] = read_json(self.cache_path, {}) or {}

    def expand(self, parties: list[dict[str, str]]) -> list[dict[str, Any]]:
        if self.mode == "off":
            return []
        results, missing = [], []
        for party in parties:
            key = self._key(party)
            if key in self.cache:
                results.append(self.cache[key])
            else:
                missing.append(party)
        if self.mode != "azure" or not missing:
            return results
        batch_size = max(1, int(self.config.get("batch_size", 10)))
        batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
        workers = max(1, int(self.config.get("workers", 4)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._call_azure, batch): batch for batch in batches}
            for future in as_completed(futures):
                entities = future.result()
                for entity in entities:
                    party = next((item for item in futures[future] if item["partyId"] == entity.get("party_id")), None)
                    if party:
                        self.cache[self._key(party)] = entity
                        results.append(entity)
        write_json(self.cache_path, self.cache)
        return results

    def _key(self, party: dict[str, str]) -> str:
        return content_hash({
            "party": party,
            "prompt": self.prompt,
            "deployment": self.config.get("deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            "version": self.config.get("version", "v1"),
            "minimum_candidate_confidence": self.config.get("minimum_candidate_confidence", 0.70),
            "minimum_parent_confidence": self.config.get("minimum_parent_confidence", 0.85),
        })

    def _call_azure(self, parties: list[dict[str, str]]) -> list[dict[str, Any]]:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' or 'all' project extra to enable Azure expansion") from exc
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        deployment = str(self.config.get("deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        if not all((endpoint, deployment, api_version, api_key)):
            raise RuntimeError("Azure OpenAI environment variables are incomplete")
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
            timeout=float(self.config.get("timeout_seconds", 30)),
            max_retries=int(self.config.get("max_retries", 2)),
        )
        # Do not expose or trust authoritative party IDs in the LLM contract.
        # A batch-local reference correlates each response; we attach the real ID below.
        parties_by_ref = {str(index): party for index, party in enumerate(parties)}
        requested = [
            {"request_ref": ref, "party_name": party["partyName"]}
            for ref, party in parties_by_ref.items()
        ]
        response = client.chat.completions.create(
            model=deployment,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps({"verified_parties": requested})},
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        entities = payload.get("entities", payload if isinstance(payload, list) else [])
        return _validate_entities(
            entities,
            parties_by_ref,
            float(self.config.get("minimum_candidate_confidence", 0.70)),
            float(self.config.get("minimum_parent_confidence", 0.85)),
        )


def _validate_entities(
    raw: object,
    parties_by_ref: dict[str, dict[str, str]],
    minimum_candidate_confidence: float = 0.70,
    minimum_parent_confidence: float = 0.85,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    output, seen_refs = [], set()
    for entity in raw:
        if not isinstance(entity, dict):
            continue
        request_ref = str(entity.get("request_ref", ""))
        party = parties_by_ref.get(request_ref)
        if not party or request_ref in seen_refs:
            continue
        seen_refs.add(request_ref)
        candidates_by_name: dict[str, dict[str, Any]] = {}
        for candidate in entity.get("match_candidates", []):
            if not isinstance(candidate, dict) or not str(candidate.get("name", "")).strip():
                continue
            name = str(candidate["name"]).strip()
            name_key = normalize_name(name)
            candidate_type = str(candidate.get("candidate_type", "")).strip()
            confidence = _probability(candidate.get("llm_confidence"))
            if (
                not name_key
                or name_key == normalize_name(party["partyName"])
                or candidate_type not in CANDIDATE_TYPES
                or confidence is None
                or confidence < minimum_candidate_confidence
            ):
                continue
            item = {
                "name": name,
                "candidate_type": candidate_type,
                "llm_confidence": confidence,
                "source": "llm",
            }
            existing = candidates_by_name.get(name_key)
            if existing is None or confidence > float(existing["llm_confidence"]):
                candidates_by_name[name_key] = item
        parent = entity.get("suggested_parent")
        parent_confidence = _probability(parent.get("llm_confidence")) if isinstance(parent, dict) else None
        if (
            not isinstance(parent, dict)
            or not str(parent.get("name", "")).strip()
            or normalize_name(parent.get("name")) == normalize_name(party["partyName"])
            or parent_confidence is None
            or parent_confidence < minimum_parent_confidence
        ):
            parent = None
        else:
            parent = {
                "name": str(parent["name"]).strip(),
                "llm_confidence": parent_confidence,
                "source": "llm",
            }
            candidates_by_name.pop(normalize_name(parent["name"]), None)
        output.append({
            "party_id": party["partyId"],
            "party_name": party["partyName"],
            "match_candidates": list(candidates_by_name.values()),
            "suggested_parent": parent,
        })
    return output


def _probability(value: object) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
