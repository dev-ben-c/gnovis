#!/usr/bin/env python3
"""Terminal chat with Ollama models + Gnovis memory tools."""

import argparse
import json
import re
import sys
import requests
from ollama import chat
from engram.store import MemoryStore
from html.parser import HTMLParser

store = MemoryStore()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search Engram memories by natural language query. Use this to look up facts, preferences, episodes, infrastructure details, project context, or anything the user may have stored previously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "category": {"type": "string", "description": "Optional category filter (e.g., 'nas', 'proxmox', 'network')"},
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a new memory in Engram. Use for facts (stable knowledge), episodes (experiential context), or preferences (user prefs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store"},
                    "memory_type": {"type": "string", "enum": ["fact", "episode", "preference"], "description": "Type of memory"},
                    "category": {"type": "string", "description": "Category (e.g., 'network', 'nas', 'project')"},
                    "key": {"type": "string", "description": "Unique key within category for dedup"},
                    "context": {"type": "string", "description": "Why you're storing this"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using SearXNG. Use this for current information, facts you don't know, or anything not in Engram memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract text content from a URL. Use this to read web pages, articles, or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Get bootstrap context from Engram — preferences and recent facts. Call at the start of a conversation or when you need broad context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional topic to focus context on"},
                },
                "required": [],
            },
        },
    },
]


def handle_tool_call(name, args, model_id):
    """Execute an Engram tool and return the result."""
    if name == "recall":
        results = store.recall(
            query=args["query"],
            category=args.get("category"),
            limit=args.get("limit", 5),
            caller_model=model_id,
        )
        if hasattr(results, "all"):
            memories = results.all
        elif hasattr(results, "memories"):
            memories = results.memories
        else:
            memories = list(results) if hasattr(results, "__iter__") else [results]
        if not memories:
            return "No memories found."
        lines = []
        for m in memories:
            lines.append(f"[{m.category}/{m.key or 'no-key'}] ({m.memory_type}) {m.content[:300]}")
        return "\n\n".join(lines)

    elif name == "remember":
        mem = store.remember(
            content=args["content"],
            memory_type=args.get("memory_type", "fact"),
            category=args.get("category", "general"),
            key=args.get("key"),
            context=args.get("context"),
            model=model_id,
        )
        return f"Stored: [{mem.category}/{mem.key}] {mem.content[:100]}"

    elif name == "get_context":
        ctx = store.get_context(
            topic=args.get("topic"),
            caller_model=model_id,
        )
        parts = []
        if ctx.get("preferences"):
            parts.append("PREFERENCES:\n" + "\n".join(f"- {p['content']}" for p in ctx["preferences"]))
        if ctx.get("recent"):
            parts.append("RECENT FACTS:\n" + "\n".join(f"- [{r['category']}] {r['content'][:200]}" for r in ctx["recent"]))
        return "\n\n".join(parts) if parts else "No context available."

    elif name == "web_search":
        try:
            resp = requests.get(
                "http://192.168.0.69:8080/search",
                params={"q": args["query"], "format": "json"},
                timeout=15,
            )
            data = resp.json()
            results = data.get("results", [])[:args.get("max_results", 5)]
            if not results:
                return "No search results found."
            lines = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("content", "")[:200]
                lines.append(f"**{title}**\n{url}\n{snippet}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"

    elif name == "fetch_url":
        try:
            resp = requests.get(args["url"], timeout=15, headers={"User-Agent": "engram-chat/1.0"})
            resp.raise_for_status()
            # Strip HTML tags to get plain text
            text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:3000]
        except Exception as e:
            return f"Fetch error: {e}"

    return f"Unknown tool: {name}"


def main():
    parser = argparse.ArgumentParser(description="Chat with Ollama + Engram memory")
    parser.add_argument("model", nargs="?", default="qwen3.5:35b-a3b", help="Ollama model name")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    args = parser.parse_args()

    model_id = args.model.split(":")[0]  # e.g., 'qwen3.5'
    print(f"\033[1;32mEngram Chat\033[0m — {args.model} + memory & web tools")
    print(f"Type /quit to exit, /ctx [topic] for context bootstrap\n")

    system_msg = (
        "You are a helpful assistant with access to tools:\n"
        "- recall: Search Engram persistent memory for stored facts, preferences, infrastructure details, project context.\n"
        "- remember: Store new facts, episodes, or preferences to Engram.\n"
        "- get_context: Load preferences and recent facts from Engram.\n"
        "- web_search: Search the web via SearXNG for current information.\n"
        "- fetch_url: Fetch and read web page content.\n\n"
        "Use recall before answering questions about the user's infrastructure, projects, or past work. "
        "Use web_search for current events, external facts, or anything not in memory. "
        "Always include your model name when storing memories."
    )

    messages = [{"role": "system", "content": system_msg}]

    # Options for thinking mode
    options = {}
    if args.no_think:
        options["think"] = False

    while True:
        try:
            user_input = input("\033[1;36myou>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input.strip():
            continue
        if user_input.strip() == "/quit":
            print("Bye!")
            break
        if user_input.strip().startswith("/ctx"):
            topic = user_input.strip()[4:].strip() or None
            ctx = handle_tool_call("get_context", {"topic": topic}, model_id)
            print(f"\033[1;33m[context]\033[0m\n{ctx}\n")
            messages.append({"role": "system", "content": f"Context loaded:\n{ctx}"})
            continue

        messages.append({"role": "user", "content": user_input})

        # Tool-calling loop
        while True:
            response = chat(
                model=args.model,
                messages=messages,
                tools=TOOLS,
                options=options if options else None,
            )

            msg = response.message

            if msg.tool_calls:
                # Add assistant message with tool calls
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments)

                    print(f"\033[1;33m[{fn_name}]\033[0m {json.dumps(fn_args, indent=None)}")
                    result = handle_tool_call(fn_name, fn_args, model_id)
                    print(f"\033[0;33m{result[:300]}\033[0m\n")

                    messages.append({"role": "tool", "content": result})

                # Loop back to get the final response
                continue
            else:
                # Final text response
                content = msg.content or ""
                # Strip thinking tags if present
                if "<think>" in content:
                    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

                messages.append({"role": "assistant", "content": content})
                print(f"\033[1;32m{model_id}>\033[0m {content}\n")
                break


if __name__ == "__main__":
    main()
