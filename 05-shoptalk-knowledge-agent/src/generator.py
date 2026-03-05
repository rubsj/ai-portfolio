"""LLM generation with citation extraction via LiteLLM.

LiteLLMClient wraps any provider (OpenAI, Anthropic, Cohere) behind one
interface. Default: gpt-4o-mini for generation, gpt-4o for evaluation.

Citation extraction: regex parses [N] markers from answers, validates N is
within chunk range, maps to Citation objects. Parse-only — LLM-as-Judge
handles semantic citation quality.

LiteLLM over raw OpenAI SDK: multi-provider support per spec. Minimal code
difference, but lets us swap providers without touching call sites.
"""