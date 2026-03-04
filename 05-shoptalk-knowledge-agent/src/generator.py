"""LLM generation with citation extraction via LiteLLM.

LiteLLMClient implements BaseLLM and wraps any provider (OpenAI, Anthropic,
Cohere) behind one interface. Default: OpenAI gpt-4o-mini for generation,
gpt-4o for evaluation.

Citation extraction: regex parses [N] markers from generated answers, validates
N is within the provided chunk range, and maps to Citation objects. Parse-only
validation — LLM-as-Judge handles semantic citation quality.

Why LiteLLM over raw OpenAI SDK: multi-provider adapter pattern. Satisfies spec
requirement. Minimal code difference but teaches provider flexibility.
"""
