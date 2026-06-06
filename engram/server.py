"""Gnovis MCP server — persistent memory for LLMs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .store import MemoryStore, RecallResult, DuplicateMemoryError, Challenge

logger = logging.getLogger("engram")

DB_PATH = os.environ.get("ENGRAM_DB", str(Path.home() / ".engram" / "memory.db"))
store = MemoryStore(DB_PATH)
server = Server("engram")


def _fmt_memory(m, provenance: str | None = None) -> str:
    """Format a memory for display.

    provenance: "own" = caller's prior memory (suppress model/context for continuity),
                "other" = different model's conclusion (show attribution),
                None/"unknown" = current behavior (show everything).
    """
    parts = [f"[{m.id}] ({m.memory_type}/{m.category}"]
    if m.key:
        parts[0] += f"/{m.key}"
    parts[0] += f") confidence={m.confidence:.1f}"
    if m.score > 0:
        parts[0] += f" score={m.score:.3f}"

    if provenance == "own":
        # Caller's own prior memory — show content directly, no model/context clutter
        parts.append(m.content)
    elif provenance == "other":
        # Another model's conclusion — flag with attribution
        parts.append(f"[{m.model} concluded]: {m.content}")
        if m.context:
            parts.append(f"  their reasoning: {m.context}")
    else:
        # Unknown provenance or no caller_model — show everything (current behavior)
        parts.append(m.content)
        if m.model:
            parts.append(f"model: {m.model}")
        if m.context:
            parts.append(f"context: {m.context}")

    if m.tags:
        parts.append(f"tags: {', '.join(m.tags)}")
    if getattr(m, "host", None):
        parts.append(f"host: {m.host}")
    parts.append(f"updated: {m.updated_at} | accessed: {m.accessed_at} ({m.access_count}x)")
    return "\n".join(parts)


def _fmt_history(h) -> str:
    """Format a history entry for display."""
    parts = [f"[{h.created_at}] {h.action}"]
    if h.model:
        parts[0] += f" by {h.model}"
    if h.action == "updated" and h.old_content != h.new_content:
        parts.append(f"  was: {h.old_content[:200] if h.old_content else '(none)'}")
        parts.append(f"  now: {h.new_content[:200] if h.new_content else '(none)'}")
    elif h.action == "created":
        parts.append(f"  content: {h.new_content[:200] if h.new_content else '(none)'}")
    elif h.action == "forgotten":
        parts.append(f"  deleted: {h.old_content[:200] if h.old_content else '(none)'}")
    if h.old_confidence != h.new_confidence and h.old_confidence is not None:
        parts.append(f"  confidence: {h.old_confidence} -> {h.new_confidence}")
    if h.context:
        parts.append(f"  reasoning: {h.context}")
    return "\n".join(parts)


def _fmt_relationship(r) -> str:
    """Format a relationship for display."""
    line = f"[{r.id}] {r.entity_from} --({r.relation_type})--> {r.entity_to}"
    if r.metadata:
        line += f" {json.dumps(r.metadata)}"
    return line


def _fmt_challenge(c: Challenge) -> str:
    """Format a challenge with its arguments for display."""
    parts = [f"[{c.id}] Challenge on memory {c.target_memory_id}"]
    parts.append(f"  Status: {c.status} | Started by: {c.challenger_model} | {c.created_at}")
    if c.resolution:
        parts.append(f"  Resolution ({c.resolved_by}): {c.resolution}")
    parts.append(f"  Arguments ({len(c.arguments)}):")
    for a in c.arguments:
        evidence_str = f" [cites: {', '.join(a.evidence)}]" if a.evidence else ""
        parts.append(f"    [{a.position}] {a.model}: {a.argument}{evidence_str}")
    return "\n".join(parts)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="remember",
            description=(
                "Store a new memory. Use memory_type='fact' for stable knowledge (configs, IPs, "
                "architecture), 'episode' for experiential context (debugging sessions, decisions), "
                "'preference' for user preferences, 'diary' for private reflections. "
                "Memories with the same category+key+model are auto-upserted. "
                "Different models can store different values for the same category+key."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["fact", "episode", "preference", "diary"],
                        "default": "fact",
                        "description": "Type: fact (stable knowledge), episode (experiential), preference (user prefs), diary (private reflections)",
                    },
                    "category": {
                        "type": "string",
                        "default": "general",
                        "description": "Category for organization (e.g., 'network', 'nas', 'proxmox', 'project')",
                    },
                    "key": {
                        "type": "string",
                        "description": "Unique key within category (for facts). Same category+key = upsert.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for filtering",
                    },
                    "confidence": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Confidence level 0.0-1.0",
                    },
                    "source": {
                        "type": "string",
                        "description": "Where this was learned (e.g., 'user stated', 'observed in logs')",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model identifier (e.g., 'claude-opus-4-6', 'qwen3:32b'). Self-identify here.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Reasoning or justification for storing this memory. Explain WHY you concluded this.",
                    },
                    "host": {
                        "type": "string",
                        "description": "Hostname this memory applies to (e.g. 'rabidllm', 'RabidNAS', 'pve'). Omit for host-agnostic facts. Use this when a fact is system-specific so it won't bleed across machines.",
                    },
                },
                "required": ["content", "model"],
            },
        ),
        types.Tool(
            name="recall",
            description=(
                "Search memories by natural language query. Returns ranked results combining "
                "keyword relevance, recency, access frequency, and confidence. This is your "
                "primary retrieval tool — use it to find what you know about a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter to a specific category",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["fact", "episode", "preference", "diary"],
                        "description": "Filter to a specific memory type",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to memories with any of these tags",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max results to return",
                    },
                    "caller_model": {
                        "type": "string",
                        "description": "Your model identifier (e.g., 'claude-opus-4-6'). "
                                       "When provided, your own prior memories appear as continuity, "
                                       "while other models' memories are flagged with attribution.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "own"],
                        "default": "all",
                        "description": "Scope: 'all' (default) returns all models' memories; "
                                       "'own' filters to only your model family's memories.",
                    },
                    "host": {
                        "type": "string",
                        "description": "Filter to memories for this host (host-agnostic memories, with NULL host, are always included).",
                    },
                    "caller_host": {
                        "type": "string",
                        "description": "Your hostname (e.g. 'rabidllm'). When provided, memories matching this host are boosted in ranking and memories with a different host are penalized.",
                    },
                    "before": {
                        "type": "string",
                        "description": "ISO 8601 date — only return memories created before this date. "
                                       "Useful for temporal queries like 'what happened last week.'",
                    },
                    "after": {
                        "type": "string",
                        "description": "ISO 8601 date — only return memories created after this date.",
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "L2 distance threshold for abstention. If no memory is closer "
                                       "than this threshold, returns empty results instead of irrelevant "
                                       "matches. Lower = stricter. Typical range: 0.8 (strict) to 1.2 (lenient).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="forget",
            description="Delete a memory by ID, or by category+key for facts. Model must own the memory. Records provenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to delete",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category (used with key for fact deletion, scoped to your model)",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key (used with category for fact deletion)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model performing the deletion (must own the memory)",
                    },
                    "context": {
                        "type": "string",
                        "description": "Why this memory is being deleted",
                    },
                },
                "required": ["model"],
            },
        ),
        types.Tool(
            name="update",
            description="Update specific fields of an existing memory. Model must own the memory. Records provenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to update",
                    },
                    "content": {"type": "string"},
                    "category": {"type": "string"},
                    "key": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "model": {
                        "type": "string",
                        "description": "Model performing the update (must own the memory)",
                    },
                    "context": {
                        "type": "string",
                        "description": "Reasoning for the update — especially important if changing a conclusion",
                    },
                    "host": {
                        "type": "string",
                        "description": "Hostname this memory applies to. Set to bind a memory to a specific host, or omit to leave unchanged.",
                    },
                },
                "required": ["memory_id", "model"],
            },
        ),
        types.Tool(
            name="relate",
            description=(
                "Create or update a relationship between two entities. Useful for tracking "
                "how services, machines, projects, and concepts connect to each other."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_from": {
                        "type": "string",
                        "description": "Source entity (e.g., 'Immich')",
                    },
                    "entity_to": {
                        "type": "string",
                        "description": "Target entity (e.g., 'TrueNAS')",
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Relationship type (e.g., 'runs_on', 'depends_on', 'connects_to')",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata about the relationship",
                    },
                    "confidence": {
                        "type": "number",
                        "default": 1.0,
                    },
                },
                "required": ["entity_from", "entity_to", "relation_type"],
            },
        ),
        types.Tool(
            name="get_entity",
            description="Get all memories and relationships for a named entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to look up",
                    },
                },
                "required": ["entity"],
            },
        ),
        types.Tool(
            name="get_context",
            description=(
                "Bootstrap context for a new conversation. Returns user preferences, "
                "recently updated facts, and optionally topic-specific memories. "
                "Call this at the start of a conversation to prime your memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic to focus context retrieval on",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                    "caller_model": {
                        "type": "string",
                        "description": "Your model identifier (e.g., 'claude-opus-4-6'). "
                                       "When provided, memories are annotated with provenance "
                                       "(own/other/unknown) so you know their source.",
                    },
                    "caller_host": {
                        "type": "string",
                        "description": "Your hostname (e.g. 'rabidllm'). When provided, topic-specific memory ranking boosts host matches and penalizes mismatches.",
                    },
                },
            },
        ),
        types.Tool(
            name="history",
            description=(
                "View the provenance/audit trail for a memory. Shows who created or modified it, "
                "what changed, and their reasoning. Essential for understanding divergent conclusions "
                "from different models. Can also filter by model to see all contributions from a specific model."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to get history for",
                    },
                    "model": {
                        "type": "string",
                        "description": "Filter history by model (shows all that model's changes across all memories)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
            },
        ),
        types.Tool(
            name="list_categories",
            description="List all memory categories with counts.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="stats",
            description="Get memory system statistics: total count, types, staleness, DB path.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_stale",
            description="Find least-active memories using ACT-R power-law decay. "
                        "Ranks by activation = ln(access_count+1) - 0.5·ln(days_since_access+1). "
                        "Lowest activation = most forgotten. Pre-filters to memories not accessed in N days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Days since last access to consider stale",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
            },
        ),
        types.Tool(
            name="backfill_embeddings",
            description=(
                "Generate vector embeddings for all memories that don't have them yet. "
                "Run this once after enabling hybrid search, or after changing the embed model. "
                "Uses nomic-embed-text via Ollama. Returns counts of embedded/failed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "default": 50,
                        "description": "Number of memories to embed per batch commit",
                    },
                },
            },
        ),
        types.Tool(
            name="list_models",
            description="List all models that have stored memories, with counts and latest activity.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="find_disagreements",
            description=(
                "Find category+key pairs where multiple models have stored different memories. "
                "Useful for identifying topics where models disagree."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="challenge",
            description=(
                "Start a debate about an existing memory. Challenge a memory you believe is "
                "wrong, incomplete, or misleading. Provide your counter-argument and optionally "
                "cite other memory IDs as evidence. Other models can then respond via 'debate'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target_memory_id": {
                        "type": "string",
                        "description": "ID of the memory being challenged",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model making the challenge (self-identify)",
                    },
                    "argument": {
                        "type": "string",
                        "description": "Your counter-argument: why this memory is wrong/incomplete",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory IDs cited as supporting evidence",
                    },
                },
                "required": ["target_memory_id", "model", "argument"],
            },
        ),
        types.Tool(
            name="debate",
            description=(
                "Add an argument to an existing challenge. Any model can participate. "
                "Use position='defense' to support the target memory, 'rebuttal' to "
                "support the challenger, or 'synthesis' to propose a merged view."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "challenge_id": {
                        "type": "string",
                        "description": "ID of the challenge to respond to",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model making the argument (self-identify)",
                    },
                    "argument": {
                        "type": "string",
                        "description": "Your argument",
                    },
                    "position": {
                        "type": "string",
                        "enum": ["defense", "rebuttal", "synthesis"],
                        "default": "rebuttal",
                        "description": "Your stance: defense (support target), rebuttal (support challenger), synthesis (merged view)",
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Memory IDs cited as supporting evidence",
                    },
                },
                "required": ["challenge_id", "model", "argument"],
            },
        ),
        types.Tool(
            name="get_challenges",
            description="List challenges/debates. Filter by status ('open', 'accepted', 'rejected', 'synthesized') or target memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "accepted", "rejected", "synthesized"],
                        "default": "open",
                        "description": "Filter by status (default: open)",
                    },
                    "target_memory_id": {
                        "type": "string",
                        "description": "Filter to challenges on a specific memory",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
            },
        ),
        types.Tool(
            name="resolve_challenge",
            description=(
                "Resolve a challenge. 'accepted' = target memory was wrong (challenger wins), "
                "'rejected' = challenge was wrong (target stands), 'synthesized' = both had "
                "partial truth. Resolution does NOT auto-modify memories — use update/remember "
                "separately to apply the conclusion."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "challenge_id": {
                        "type": "string",
                        "description": "ID of the challenge to resolve",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model (or 'user') resolving the challenge",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["accepted", "rejected", "synthesized"],
                        "description": "Resolution: accepted (target wrong), rejected (challenge wrong), synthesized (merged)",
                    },
                    "resolution": {
                        "type": "string",
                        "description": "Explanation of the resolution and reasoning",
                    },
                },
                "required": ["challenge_id", "model", "status", "resolution"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [types.TextContent(type="text", text=f"Error: {e}")]


def _dispatch(name: str, args: dict) -> str:
    if name == "remember":
        try:
            m = store.remember(
                content=args["content"],
                memory_type=args.get("memory_type", "fact"),
                category=args.get("category", "general"),
                key=args.get("key"),
                tags=args.get("tags"),
                confidence=args.get("confidence", 1.0),
                source=args.get("source"),
                model=args["model"],
                context=args.get("context"),
                host=args.get("host"),
            )
        except DuplicateMemoryError as e:
            return (
                f"Duplicate detected (~{e.similarity:.0%} similar to existing memory). "
                f"Not stored.\n\nExisting memory:\n{_fmt_memory(e.existing)}\n\n"
                f"Use 'update' with memory_id={e.existing.id!r} to modify it, "
                f"or use a different category/key if this is intentionally distinct."
            )
        return f"Stored:\n{_fmt_memory(m)}"

    elif name == "recall":
        result = store.recall(
            query=args["query"],
            category=args.get("category"),
            memory_type=args.get("memory_type"),
            tags=args.get("tags"),
            limit=args.get("limit", 10),
            caller_model=args.get("caller_model"),
            scope=args.get("scope", "all"),
            host=args.get("host"),
            caller_host=args.get("caller_host"),
            before=args.get("before"),
            after=args.get("after"),
            min_similarity=args.get("min_similarity"),
        )
        if isinstance(result, RecallResult):
            if result.total == 0:
                return "No memories found matching that query."
            lines = [f"Found {result.total} memories ({len(result.own)} yours, "
                     f"{len(result.others)} from other models, {len(result.unknown)} unattributed):\n"]
            for m in result.own:
                lines.append(_fmt_memory(m, provenance="own"))
                lines.append("")
            for m in result.others:
                lines.append(_fmt_memory(m, provenance="other"))
                lines.append("")
            for m in result.unknown:
                lines.append(_fmt_memory(m, provenance="unknown"))
                lines.append("")
            return "\n".join(lines)
        else:
            # Backward compat: list[Memory]
            if not result:
                return "No memories found matching that query."
            lines = [f"Found {len(result)} memories:\n"]
            for m in result:
                lines.append(_fmt_memory(m))
                lines.append("")
            return "\n".join(lines)

    elif name == "forget":
        model = args["model"]
        context = args.get("context")
        try:
            if args.get("memory_id"):
                ok = store.forget(args["memory_id"], model=model, context=context)
            elif args.get("category") and args.get("key"):
                ok = store.forget_by_key(args["category"], args["key"], model=model, context=context)
            else:
                return "Provide either memory_id or both category+key."
        except PermissionError as e:
            return f"Permission denied: {e}"
        return "Forgotten." if ok else "Memory not found."

    elif name == "update":
        try:
            m = store.update(
                memory_id=args["memory_id"],
                content=args.get("content"),
                category=args.get("category"),
                key=args.get("key"),
                tags=args.get("tags"),
                confidence=args.get("confidence"),
                model=args["model"],
                context=args.get("context"),
                host=args.get("host"),
            )
        except PermissionError as e:
            return f"Permission denied: {e}"
        if not m:
            return "Memory not found."
        return f"Updated:\n{_fmt_memory(m)}"

    elif name == "relate":
        r = store.relate(
            entity_from=args["entity_from"],
            entity_to=args["entity_to"],
            relation_type=args["relation_type"],
            metadata=args.get("metadata"),
            confidence=args.get("confidence", 1.0),
        )
        return f"Related:\n{_fmt_relationship(r)}"

    elif name == "get_entity":
        data = store.get_entity(args["entity"])
        parts = [f"Entity: {data['entity']}\n"]
        if data["memories"]:
            parts.append(f"Memories ({len(data['memories'])}):")
            for m_dict in data["memories"]:
                m = type("M", (), m_dict)()  # quick attr access
                parts.append(f"  [{m_dict['id']}] {m_dict['content'][:120]}")
        if data["relationships"]:
            parts.append(f"\nRelationships ({len(data['relationships'])}):")
            for r_dict in data["relationships"]:
                parts.append(f"  {r_dict['entity_from']} --({r_dict['relation_type']})--> {r_dict['entity_to']}")
        if not data["memories"] and not data["relationships"]:
            parts.append("No memories or relationships found.")
        return "\n".join(parts)

    elif name == "get_context":
        caller_model = args.get("caller_model")
        ctx = store.get_context(
            topic=args.get("topic"),
            limit=args.get("limit", 20),
            caller_model=caller_model,
            caller_host=args.get("caller_host"),
        )
        parts = []
        if ctx["preferences"]:
            parts.append(f"Preferences ({len(ctx['preferences'])}):")
            for p in ctx["preferences"]:
                prov = p.get("provenance")
                prefix = f"({p['model']} set) " if prov == "other" and p.get("model") else ""
                parts.append(f"  - {prefix}{p['content'][:120]}")
        if ctx["recent"]:
            parts.append(f"\nRecent facts ({len(ctx['recent'])}):")
            for f in ctx["recent"]:
                prov = f.get("provenance")
                attr = f" (per {f['model']})" if prov == "other" and f.get("model") else ""
                parts.append(f"  [{f['category']}/{f.get('key', '?')}] {f['content'][:120]}{attr}")
        if ctx.get("topic_memories"):
            parts.append(f"\nTopic memories ({len(ctx['topic_memories'])}):")
            for m in ctx["topic_memories"]:
                prov = m.get("provenance")
                attr = f" (per {m['model']})" if prov == "other" and m.get("model") else ""
                parts.append(f"  [{m['id']}] {m['content'][:120]}{attr}")
        if not parts:
            parts.append("Memory is empty. Start building it with the 'remember' tool.")
        return "\n".join(parts)

    elif name == "history":
        if args.get("memory_id"):
            entries = store.get_history(args["memory_id"])
            if not entries:
                return "No history found for this memory."
            lines = [f"History for memory {args['memory_id']} ({len(entries)} entries):\n"]
            for h in entries:
                lines.append(_fmt_history(h))
                lines.append("")
            return "\n".join(lines)
        elif args.get("model"):
            entries = store.get_history_by_model(args["model"], limit=args.get("limit", 20))
            if not entries:
                return f"No history found for model {args['model']}."
            lines = [f"History from {args['model']} ({len(entries)} entries):\n"]
            for h in entries:
                lines.append(f"memory:{h.memory_id} — {_fmt_history(h)}")
                lines.append("")
            return "\n".join(lines)
        else:
            return "Provide either memory_id or model to query history."

    elif name == "list_categories":
        cats = store.list_categories()
        if not cats:
            return "No categories yet."
        lines = ["Categories:"]
        for c in cats:
            lines.append(f"  {c['category']} ({c['memory_type']}): {c['count']}")
        return "\n".join(lines)

    elif name == "stats":
        s = store.stats()
        return json.dumps(s, indent=2)

    elif name == "get_stale":
        memories = store.get_stale(
            days=args.get("days", 30),
            limit=args.get("limit", 20),
        )
        if not memories:
            return "No stale memories found."
        lines = [f"Stale memories ({len(memories)}), ranked by ACT-R activation (lowest = most forgotten):\n"]
        for m in memories:
            lines.append(f"[{m.id}] ({m.category}) activation={m.score:.2f} "
                         f"accessed={m.access_count}x last={m.accessed_at}")
            lines.append(f"  {m.content[:120]}")
            lines.append("")
        return "\n".join(lines)

    elif name == "backfill_embeddings":
        result = store.backfill_embeddings(batch_size=args.get("batch_size", 50))
        return json.dumps(result, indent=2)

    elif name == "list_models":
        models = store.list_models()
        if not models:
            return "No models have stored memories yet."
        lines = ["Models:"]
        for m in models:
            lines.append(
                f"  {m['model']}: {m['count']} memories, "
                f"latest: {m['latest_update']}, first: {m['first_memory']}"
            )
        return "\n".join(lines)

    elif name == "find_disagreements":
        disagreements = store.find_disagreements()
        if not disagreements:
            return "No disagreements found — no category+key has memories from multiple models."
        lines = [f"Found {len(disagreements)} disagreements:\n"]
        for d in disagreements:
            lines.append(f"  {d['category']}/{d['key']}: {d['model_count']} models ({d['models']})")
        return "\n".join(lines)

    elif name == "challenge":
        try:
            c = store.challenge(
                target_memory_id=args["target_memory_id"],
                model=args["model"],
                argument=args["argument"],
                evidence=args.get("evidence"),
            )
        except ValueError as e:
            return f"Error: {e}"
        # Also show the target memory for context
        target = store.recall_by_id(args["target_memory_id"])
        parts = ["Challenge created:\n"]
        if target:
            parts.append(f"Target memory:\n{_fmt_memory(target)}\n")
        parts.append(_fmt_challenge(c))
        return "\n".join(parts)

    elif name == "debate":
        try:
            c = store.debate(
                challenge_id=args["challenge_id"],
                model=args["model"],
                argument=args["argument"],
                position=args.get("position", "rebuttal"),
                evidence=args.get("evidence"),
            )
        except ValueError as e:
            return f"Error: {e}"
        return f"Argument added:\n\n{_fmt_challenge(c)}"

    elif name == "get_challenges":
        challenges = store.get_challenges(
            status=args.get("status", "open"),
            target_memory_id=args.get("target_memory_id"),
            limit=args.get("limit", 20),
        )
        if not challenges:
            status = args.get("status", "open")
            return f"No {status} challenges found."
        lines = [f"Found {len(challenges)} challenge(s):\n"]
        for c in challenges:
            lines.append(_fmt_challenge(c))
            lines.append("")
        return "\n".join(lines)

    elif name == "resolve_challenge":
        try:
            c = store.resolve_challenge(
                challenge_id=args["challenge_id"],
                status=args["status"],
                resolution=args["resolution"],
                resolved_by=args["model"],
            )
        except ValueError as e:
            return f"Error: {e}"
        return f"Challenge resolved:\n\n{_fmt_challenge(c)}"

    else:
        return f"Unknown tool: {name}"


def main():
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Engram MCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode: stdio (default) or sse (HTTP/SSE for remote access)",
    )
    parser.add_argument(
        "--port", type=int, default=8093,
        help="Port for SSE transport (default: 8093)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind SSE server (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Engram starting with DB at {DB_PATH}")

    if args.transport == "sse":
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from mcp.server.sse import SseServerTransport
        import uvicorn

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
            return Response()

        async def health(request: Request):
            return JSONResponse({"status": "ok", "db": DB_PATH})

        async def recall_http(request: Request):
            q = (request.query_params.get("q") or "").strip()
            if not q:
                return JSONResponse({"count": 0, "memories": []})
            try:
                limit = int(request.query_params.get("limit", "5"))
            except (TypeError, ValueError):
                limit = 5
            caller_model = request.query_params.get("caller_model")
            caller_host = request.query_params.get("caller_host")
            try:
                result = store.recall(query=q, limit=limit, caller_model=caller_model, caller_host=caller_host)
            except Exception as e:
                return JSONResponse({"count": 0, "memories": [], "error": str(e)}, status_code=500)
            mems = []
            def _add(m, provenance):
                mems.append({"id": m.id, "memory_type": m.memory_type, "category": m.category, "key": m.key, "model": m.model, "score": getattr(m, "score", 0.0), "content": m.content, "provenance": provenance})
            if isinstance(result, RecallResult):
                for m in result.own: _add(m, "own")
                for m in result.others: _add(m, "other")
                for m in result.unknown: _add(m, "unknown")
            else:
                for m in (result or []): _add(m, "unknown")
            mems.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
            return JSONResponse({"count": len(mems), "memories": mems})

        app = Starlette(
            routes=[
                Route("/health", health),
                Route("/recall", recall_http),
                Route("/sse", handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        logger.info(f"Engram SSE server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    else:
        async def _run():
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )

        asyncio.run(_run())


if __name__ == "__main__":
    main()
