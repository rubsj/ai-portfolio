"""
=============================================================================
PYTHON PRIMER — For Java/TypeScript Developers
=============================================================================

Run this file section by section in VS Code with Claude Code.
Each section has:
  1. CONCEPT — what it is, mapped to Java/TS
  2. EXAMPLE — working code to read and run
  3. EXERCISE — break it, fix it, extend it
  4. WHY THIS MATTERS FOR P1 — how you'll use it tomorrow

HOW TO USE:
  - Open this file in VS Code
  - Use Claude Code in the terminal: `uv run python python_primer.py`
  - Or run sections interactively: `uv run python -i python_primer.py`
  - Try each exercise. When stuck, ask Claude Code: "explain this pattern"

=============================================================================
"""

# ============================================================================
# SECTION 1: TYPE HINTS
# ============================================================================
# Java: public String greet(String name, int age) { ... }
# TS:   function greet(name: string, age: number): string { ... }
# Python: def greet(name: str, age: int) -> str: ...
#
# KEY DIFFERENCE: Python type hints are NOT enforced at runtime by default.
# They're for: IDE autocomplete, static analysis (mypy/pyright), documentation.
# Pydantic DOES enforce them at runtime — that's why we use it.
# ============================================================================

from collections import defaultdict


def greet(name: str, age: int) -> str:
    """
    Type hints look like TS annotations but go AFTER the parameter name.

    Java:   String greet(String name, int age)
    TS:     greet(name: string, age: number): string
    Python: greet(name: str, age: int) -> str
    """
    return f"Hello {name}, you are {age} years old"


# Python 3.12 native generics — no need to import from typing
# Java:   List<String>             Map<String, Integer>
# TS:     string[]                 Record<string, number>
# Python: list[str]                dict[str, int]

def process_names(names: list[str]) -> dict[str, int]:
    """
    WHY THIS MATTERS FOR P1:
    Your Pydantic schemas use these type hints for every field.
    `tools_required: list[str]` tells Pydantic to validate that
    every item in the list is actually a string.
    """
    return {name: len(name) for name in names}


# Optional/nullable types
# Java:   @Nullable String or Optional<String>
# TS:     name: string | null  or  name?: string
# Python: name: str | None = None

def find_user(user_id: int, nickname: str | None = None) -> dict[str, str | int]:
    """
    `str | None` means "this can be a string OR None (Python's null)".
    The `= None` makes it optional with a default value.

    In Pydantic: Field(default=None) achieves the same thing.
    """
    result: dict[str, str | int] = {"id": user_id}
    if nickname is not None:
        result["nickname"] = nickname
    return result

print("=" * 60)
print("SECTION 1: TYPE HINTS")
print("=" * 60)
print(greet("Alice", 30))
print(process_names(["Alice", "Bob", "Charlie"]))
print(find_user(1))
print(find_user(1, "ace"))
print()

# EXERCISE 1:
# Write a function `categorize_tools` that takes a list[str] of tool names
# and returns a dict[str, list[str]] grouping them by first letter.
# Example: ["wrench", "hammer", "wire cutters"] → {"w": ["wrench", "wire cutters"], "h": ["hammer"]}
# Hint: use dict.setdefault() or defaultdict

# --- YOUR CODE HERE ---
# def categorize_tools(tools: list[str]) -> dict[str, list[str]]:
#     pass
def categorize_tools(tools: list[str]) -> dict[str, list[str]]:
    res: dict[str, list[str]] = defaultdict(list)
    for tool in tools:
        key = tool[0]
        res[key].append(tool)
    return dict(res)

print(categorize_tools(["wrench", "hammer", "wire cutters"]))


# ============================================================================
# SECTION 2: F-STRINGS AND TRIPLE QUOTES
# ============================================================================
# TS:    `Hello ${name}, you have ${items.length} items`
# Python: f"Hello {name}, you have {len(items)} items"
#
# CRITICAL FOR P1: Every LLM prompt you build uses f-strings.
# The triple-quote f-string is your prompt template builder.
# ============================================================================

name = "kitchen faucet"
category = "plumbing_repair"
difficulty = "beginner"

# Basic f-string (same concept as TS template literal)
simple = f"Fix a {name} — Category: {category}"

# Expression inside f-string (you can call functions, do math, etc.)
tools = ["wrench", "plumber's tape", "bucket"]
with_expr = f"You need {len(tools)} tools: {', '.join(tools)}"

# Triple-quoted f-string — THIS IS YOUR LLM PROMPT PATTERN
# WHY: LLM prompts are multi-line. Triple quotes preserve newlines.
prompt = f"""You are an expert home repair technician.

Generate a detailed DIY repair guide for the following:
- Problem: {name}
- Category: {category}
- Difficulty: {difficulty}

Respond in JSON format with these fields:
- question: A clear question a homeowner would ask
- answer: Detailed step-by-step answer
- tools_required: List of tools needed
- safety_info: Safety warnings
- tips: Helpful tips
"""

# GOTCHA: Curly braces in f-strings conflict with JSON!
# If you need literal { } in an f-string, double them: {{ }}

json_template = f"""{{
  "category": "{category}",
  "difficulty": "{difficulty}",
  "tools_required": ["wrench", "tape"]
}}"""
# The {{ and }} produce literal { and } in the output
# While {category} and {difficulty} get interpolated


print("=" * 60)
print("SECTION 2: F-STRINGS AND TRIPLE QUOTES")
print("=" * 60)
print(simple)
print(with_expr)
print("--- LLM PROMPT ---")
print(prompt[:200] + "...")
print("--- JSON WITH ESCAPED BRACES ---")
print(json_template)
print()

# EXERCISE 2:
# Build a prompt template function that takes category, difficulty, and
# a list of example tools, and returns a formatted prompt string.
# The prompt should include a JSON structure with {{ }} escaping.
# This is EXACTLY what you'll do in generator.py tomorrow.

# --- YOUR CODE HERE ---
def build_prompt(category: str, difficulty: str, example_tools: list[str]) -> str:
    prompt = f"""this is a simple prompt  that returns json {{
        "category": "{category}"
        "difficulty": "{difficulty}"
        "example_tools": [{', '.join(f'"{t}"' for t in example_tools)}]
    }}
    """
    return prompt

print(build_prompt("plumbing_repair" , "beginner" , ["wrench", "tape"]))


# ============================================================================
# SECTION 3: LIST AND DICT COMPREHENSIONS
# ============================================================================
# Java:   items.stream().filter(x -> x.isActive()).map(x -> x.getName()).collect(toList())
# TS:     items.filter(x => x.active).map(x => x.name)
# Python: [x.name for x in items if x.active]
#
# WHY: Python devs use comprehensions EVERYWHERE. If you write a for-loop
# to build a list, a Python reviewer will suggest a comprehension.
# ============================================================================

# List comprehension — the Python way to .map().filter()
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# TS: numbers.filter(n => n % 2 === 0).map(n => n * n)
squares_of_evens = [n * n for n in numbers if n % 2 == 0]

# Dict comprehension — build a dict in one expression
# TS: Object.fromEntries(words.map(w => [w, w.length]))
words = ["wrench", "hammer", "pliers", "screwdriver"]
word_lengths = {word: len(word) for word in words}

# Nested comprehension — flatMap equivalent
# TS: categories.flatMap(cat => cat.tools)
categories_with_tools = {
    "plumbing": ["wrench", "tape"],
    "electrical": ["multimeter", "wire stripper"],
    "general": ["hammer", "screwdriver"],
}
all_tools = [tool for tools in categories_with_tools.values() for tool in tools]

# Conditional expression in comprehension (ternary)
# TS: items.map(x => x > 5 ? "high" : "low")
labels = ["high" if n > 5 else "low" for n in numbers]

# Set comprehension — unique values
# TS: [...new Set(items.map(x => x.category))]
records = [
    {"category": "plumbing"},
    {"category": "electrical"},
    {"category": "plumbing"},
    {"category": "hvac"},
]
unique_categories = {r["category"] for r in records}


print("=" * 60)
print("SECTION 3: COMPREHENSIONS")
print("=" * 60)
print(f"Squares of evens: {squares_of_evens}")
print(f"Word lengths: {word_lengths}")
print(f"All tools (flat): {all_tools}")
print(f"Labels: {labels}")
print(f"Unique categories: {unique_categories}")
print()

# EXERCISE 3:
# Given this list of repair records, use comprehensions to:
# a) Extract all unique tools across all records
# b) Create a dict of {category: count} showing how many records per category
# c) Filter to only "beginner" difficulty records and get their titles

sample_records = [
    {"title": "Fix leaky faucet", "category": "plumbing", "difficulty": "beginner", "tools": ["wrench", "tape"]},
    {"title": "Replace outlet", "category": "electrical", "difficulty": "intermediate", "tools": ["screwdriver", "multimeter"]},
    {"title": "Unclog drain", "category": "plumbing", "difficulty": "beginner", "tools": ["plunger", "wrench"]},
    {"title": "Install dimmer", "category": "electrical", "difficulty": "advanced", "tools": ["wire stripper", "multimeter", "screwdriver"]},
    {"title": "Patch drywall", "category": "general", "difficulty": "beginner", "tools": ["putty knife", "sandpaper"]},
]

# --- YOUR CODE HERE ---
# a) all_unique_tools = ...
all_unique_tools = {tool for r in sample_records for tool in r["tools"] }
print(all_unique_tools)
# b) category_counts = ...
categories = [ r["category"] for r in sample_records]
category_counts = {cat : categories.count(cat) for cat in set(categories)}
print(category_counts)
# c) beginner_titles = ...
biginner_titles = [ r["title"] for r in sample_records if r["difficulty"] == 'beginner']
print(biginner_titles)


# ============================================================================
# SECTION 4: PYDANTIC v2 — THE MAIN EVENT
# ============================================================================
# This is the single most important section. Pydantic is your universal
# validation layer for ALL 9 projects.
#
# Java equivalent:  Bean Validation (@NotNull, @Min, @Max) + Jackson for JSON
# TS equivalent:    Zod or io-ts (runtime validation libraries)
#
# KEY INSIGHT: Python type hints are NOT enforced at runtime. Pydantic IS.
# When an LLM returns garbage JSON, Pydantic catches it. This is your
# safety net against hallucinated data.
# ============================================================================

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Literal
from enum import Enum

# --- 4a: Basic Model ---
# This is like a TypeScript interface + Zod schema combined.
# Java equivalent: a POJO with Bean Validation annotations.

class Tool(BaseModel):
    """
    A tool needed for a repair task.

    WHY BaseModel not dataclass:
    - dataclass: just organizes data (like a Java record or TS interface)
    - BaseModel: organizes + validates + generates JSON schema + serializes
    - For LLM work, you NEED the validation. Always use BaseModel.
    """
    # `str` means "must be a string". Pydantic will reject int, None, etc.
    name: str = Field(
        min_length=1,       # Can't be empty string
        max_length=100,     # Reasonable upper bound
        description="Name of the tool"  # Shows up in JSON schema → LLM sees this
    )

    # `bool` with default — same as TS `optional_flag?: boolean` with default
    is_specialized: bool = Field(
        default=False,
        description="Whether this is a specialized tool most homeowners don't own"
    )


# --- 4b: Enums and Literals ---
# Java: enum Difficulty { BEGINNER, INTERMEDIATE, ADVANCED }
# TS:   type Difficulty = "beginner" | "intermediate" | "advanced"
# Python has both options:

# Option 1: Enum (more Java-like, better for iteration)
class DifficultyEnum(str, Enum):
    """
    WHY str, Enum (multiple inheritance):
    Without `str`, Pydantic would serialize as `DifficultyEnum.BEGINNER` (ugly).
    With `str`, it serializes as "beginner" (clean JSON).
    """
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# Option 2: Literal (more TS-like, simpler for small sets)
# We'll use Literal in our schemas — it's cleaner for JSON/LLM work
DifficultyLiteral = Literal["beginner", "intermediate", "advanced"]


# --- 4c: Nested Models with Validators ---

class RepairStep(BaseModel):
    """
    One step in a repair procedure.

    WHY field_validator over @validator:
    @validator is Pydantic v1 (deprecated). field_validator is v2.
    The `mode="before"` parameter means validation runs BEFORE type coercion.
    """
    step_number: int = Field(ge=1, le=50, description="Sequential step number")
    description: str = Field(min_length=10, description="What to do in this step")
    duration_minutes: int | None = Field(
        default=None,
        ge=1,
        le=480,  # 8 hours max for a single step
        description="Estimated minutes for this step"
    )
    tools_needed: list[str] = Field(
        default_factory=list,   # WHY default_factory, not default=[]:
        # In Python, default=[] is a MUTABLE DEFAULT — a classic bug.
        # All instances would SHARE the same list object.
        # default_factory=list creates a NEW empty list for each instance.
        # Java/TS don't have this gotcha because they copy on construction.
        description="Tools needed for this specific step"
    )
    tips: str | None = Field(default=None, description="Pro tip for this step")

    @field_validator("description")
    @classmethod  # WHY @classmethod: Pydantic v2 requires it. It means the validator
                  # receives the CLASS as first arg, not the INSTANCE (instance doesn't exist yet).
    def description_must_be_actionable(cls, v: str) -> str:
        """
        Validate that the description starts with a verb (actionable step).
        This catches LLM outputs like "The faucet should be..." instead of "Turn off..."
        """
        # `v` is the raw value being validated
        if not v[0].isupper():
            raise ValueError("Step description must start with a capital letter (actionable verb)")
        return v  # Always return the value (possibly transformed)


class DIYRepairRecord(BaseModel):
    """
    Complete synthetic data record for a DIY repair task.
    This is the TOP-LEVEL model that wraps everything.
    """
    question: str = Field(min_length=10, description="Question a homeowner would ask")
    answer: str = Field(min_length=50, description="Detailed answer")
    category: Literal[
        "appliance_repair",
        "plumbing_repair",
        "electrical_repair",
        "hvac_maintenance",
        "general_home_repair",
    ]
    difficulty: Literal["beginner", "intermediate", "advanced"]
    equipment_problem: str = Field(min_length=5, description="Specific problem description")
    tools_required: list[Tool] = Field(
        min_length=1,  # Must have at least one tool
        description="Tools needed for the repair"
    )
    steps: list[RepairStep] = Field(
        min_length=1,
        description="Step-by-step repair procedure"
    )
    safety_info: str = Field(min_length=10, description="Safety warnings and precautions")
    tips: str = Field(min_length=5, description="Helpful tips for the repair")

    @model_validator(mode="after")
    def validate_steps_are_sequential(self) -> "DIYRepairRecord":
        """
        WHY model_validator vs field_validator:
        - field_validator: validates ONE field in isolation
        - model_validator: validates ACROSS fields (cross-field logic)

        mode="after" means all individual fields are already validated.
        `self` is the fully constructed model instance.

        This checks that step_numbers are sequential: 1, 2, 3, ...
        LLMs sometimes skip numbers or start at 0.
        """
        step_numbers = [s.step_number for s in self.steps]
        expected = list(range(1, len(self.steps) + 1))
        if step_numbers != expected:
            raise ValueError(
                f"Steps must be sequential starting from 1. "
                f"Got {step_numbers}, expected {expected}"
            )
        return self  # model_validator must return self


# --- 4d: Using the Models ---

print("=" * 60)
print("SECTION 4: PYDANTIC v2")
print("=" * 60)

# Creating a valid instance (like `new Tool("wrench")` in Java)
tool = Tool(name="adjustable wrench", is_specialized=False)
print(f"Valid tool: {tool}")
print(f"Tool as dict: {tool.model_dump()}")      # Java: tool.toMap()
print(f"Tool as JSON: {tool.model_dump_json()}")  # Java: objectMapper.writeValueAsString(tool)

# JSON Schema generation — THIS IS WHAT YOU SEND TO THE LLM
# It tells GPT-4o-mini exactly what structure to return
print(f"\nJSON Schema for DIYRepairRecord:")
import json
schema = DIYRepairRecord.model_json_schema()
print(json.dumps(schema, indent=2)[:500] + "\n... (truncated)")

# Validation failure — THIS IS HOW YOU CATCH LLM HALLUCINATIONS
print("\n--- Validation Failures ---")
try:
    bad_tool = Tool(name="")  # Empty name violates min_length=1
except ValidationError as e:
    print(f"Empty name error:\n{e}")

try:
    bad_step = RepairStep(
        step_number=0,        # Violates ge=1
        description="bad",    # Violates min_length=10
    )
except ValidationError as e:
    print(f"\nBad step error:\n{e}")

# Parsing JSON from LLM response — model_validate_json()
# This is THE method you'll use in generator.py tomorrow
print("\n--- Parsing LLM JSON Response ---")
llm_response = json.dumps({
    "question": "How do I fix a leaking kitchen faucet?",
    "answer": "A leaking kitchen faucet is usually caused by a worn O-ring or cartridge. Here's how to fix it step by step, which should take about 30 minutes.",
    "category": "plumbing_repair",
    "difficulty": "beginner",
    "equipment_problem": "Leaking single-handle kitchen faucet dripping from spout",
    "tools_required": [
        {"name": "adjustable wrench", "is_specialized": False},
        {"name": "hex key set", "is_specialized": False},
    ],
    "steps": [
        {"step_number": 1, "description": "Turn off the water supply valves under the sink", "duration_minutes": 2},
        {"step_number": 2, "description": "Remove the faucet handle by loosening the set screw", "duration_minutes": 5, "tools_needed": ["hex key set"]},
        {"step_number": 3, "description": "Pull out the old cartridge and take it to the hardware store for matching", "duration_minutes": 10},
    ],
    "safety_info": "Always turn off water supply before starting. Place a towel in the sink to catch small parts.",
    "tips": "Take a photo of each step during disassembly to help with reassembly."
})

# model_validate_json() = parse JSON string + validate in one step
# This is Pydantic v2's replacement for parse_raw() (v1, deprecated)
record = DIYRepairRecord.model_validate_json(llm_response)
print(f"Parsed record: {record.question}")
print(f"Category: {record.category}")
print(f"Tools: {[t.name for t in record.tools_required]}")
print(f"Steps: {len(record.steps)}")
print()

# EXERCISE 4:
# a) Create a `SafetyPrecaution` model with: description (str, min 10 chars),
#    severity (Literal["low", "medium", "high", "critical"]),
#    equipment (list[str], can be empty)
#
# b) Add a field_validator to SafetyPrecaution that ensures "critical" severity
#    items have at least one equipment item (you can't have critical safety
#    without specifying what protective gear to use)
#
# c) Try creating valid and invalid instances. See what errors Pydantic gives.
#
# d) Generate the JSON schema with model_json_schema() and examine it —
#    this is what you'll embed in your LLM prompts tomorrow.

# --- YOUR CODE HERE ---
class SafetyPrecaution(BaseModel):
    description: str = Field(min_length=10, description="Description of the safety precaution")
    severity : Literal["low", "medium", "high", "critical"]
    equipment: list[str] = Field(default_factory=list)
    
    @model_validator(mode="after")
    def validate_equipment(self):
        if self.severity == "critical" and len(self.equipment) == 0 :
            raise ValueError("For critical severity items , atleast one equipment must be present")
        return self
        
print("="*60)
# valid safety precaution model with non critical severity
safety_preac1 = SafetyPrecaution(description="this is a valid one" , severity = "low")
print(f"valid SafetyPrecaution obj {safety_preac1.model_dump_json()}")
# valid safety precaution model with  critical severity
safety_preac2 = SafetyPrecaution(description="this is a valid two" , severity = "critical" , equipment=["wrench"])
print(f"valid SafetyPrecaution obj {safety_preac2.model_dump_json()}")
# invalid safety precaution model
try :
    safety_preac3 = SafetyPrecaution(description="this is a valid two" , severity = "critical")
except ValidationError as e :
    print(f"validation error {e}")
    
schema = SafetyPrecaution.model_json_schema()
print(json.dumps(schema, indent=2))
print("="*60)
    

# ============================================================================
# SECTION 5: PYTHON GOTCHAS FOR JAVA/TS DEVELOPERS
# ============================================================================
# These are the things that WILL trip you up. Better to hit them now
# in a practice file than in the middle of P1 tomorrow.
# ============================================================================

print("=" * 60)
print("SECTION 5: PYTHON GOTCHAS")
print("=" * 60)

# --- 5a: Mutable Default Arguments (THE classic Python bug) ---
# In Java/TS, default parameter values are copied per call. Not in Python.

def bad_append(item: str, items: list[str] = []) -> list[str]:
    """
    BUG: The default [] is created ONCE when the function is defined.
    Every call that uses the default SHARES the same list object.
    This is Python's most infamous gotcha.
    """
    items.append(item)
    return items

def good_append(item: str, items: list[str] | None = None) -> list[str]:
    """
    FIX: Use None as default, create a new list inside the function.
    This is the idiomatic Python pattern. You'll see it everywhere.
    """
    if items is None:
        items = []
    items.append(item)
    return items

# Demonstrate the bug
print("--- Mutable Default Bug ---")
print(f"bad_append call 1: {bad_append('a')}")  # ['a']
print(f"bad_append call 2: {bad_append('b')}")  # ['a', 'b'] — BUG! Expected ['b']
print(f"good_append call 1: {good_append('a')}")  # ['a']
print(f"good_append call 2: {good_append('b')}")  # ['b'] — Correct!

# In Pydantic, you use default_factory=list instead of default=[]
# We saw this in RepairStep.tools_needed above.


# --- 5b: `if __name__ == "__main__":` guard ---
# Java:   public static void main(String[] args)
# TS:     if (require.main === module)  (Node.js)
# Python: if __name__ == "__main__":
#
# WHY: When you `import` a Python file, ALL top-level code runs.
# This guard prevents that — code inside only runs when you execute
# the file directly, not when it's imported as a module.
#
# You'll need this in generator.py, validator.py, etc.

def main() -> None:
    print("\n--- Running as main script ---")
    print("In generator.py, this is where you'd call generate_batch()")


# --- 5c: Dictionary operations you'll use constantly ---

# .get() with default — safe access (like Optional.orElse in Java)
config = {"model": "gpt-4o-mini", "temperature": 0.7}
model = config.get("model", "gpt-3.5-turbo")      # Returns "gpt-4o-mini"
max_tokens = config.get("max_tokens", 1000)         # Returns 1000 (key doesn't exist)
# TS equivalent: config.model ?? "gpt-3.5-turbo"

# .setdefault() — get or set (no TS/Java equivalent, very useful)
cache: dict[str, str] = {}
cache.setdefault("prompt_hash_abc", "cached_response_here")
print(f"\n--- Dict operations ---")
print(f"cache after setdefault: {cache}")

# dict unpacking — merging dicts (like spread operator in TS)
# TS:    { ...defaults, ...overrides }
# Python: {**defaults, **overrides}
defaults = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000}
overrides = {"temperature": 0.9, "max_tokens": 2000}
final_config = {**defaults, **overrides}
print(f"Merged config: {final_config}")


# --- 5d: pathlib.Path — modern file handling ---
# Java:   Paths.get("data", "cache", "file.json")
# TS:     path.join("data", "cache", "file.json")
# Python: Path("data") / "cache" / "file.json"  ← uses / operator!

from pathlib import Path

cache_dir = Path("data") / "cache"
cache_file = cache_dir / "responses.json"
print(f"\n--- pathlib ---")
print(f"Cache dir: {cache_dir}")
print(f"Cache file: {cache_file}")
print(f"File stem: {cache_file.stem}")        # "responses" (no extension)
print(f"File suffix: {cache_file.suffix}")     # ".json"
print(f"Exists? {cache_file.exists()}")        # False (we haven't created it)

# Creating directories + writing files
# cache_dir.mkdir(parents=True, exist_ok=True)  # Like mkdir -p
# cache_file.write_text(json.dumps({"key": "value"}))  # Atomic write


# --- 5e: Enumerate — indexed iteration ---
# Java:   for (int i = 0; i < items.size(); i++) { ... items.get(i) ... }
# TS:     items.forEach((item, index) => { ... })
# Python: for i, item in enumerate(items):

steps = ["Turn off water", "Remove handle", "Replace cartridge"]
print(f"\n--- enumerate ---")
for i, step in enumerate(steps, start=1):  # start=1 for 1-indexed
    print(f"  Step {i}: {step}")


print()

# ============================================================================
# SECTION 6: HASHLIB — FOR LLM RESPONSE CACHING
# ============================================================================
# You'll use this tomorrow to cache LLM responses and avoid re-calling
# the API during development.
# ============================================================================

import hashlib

print("=" * 60)
print("SECTION 6: HASHLIB FOR CACHING")
print("=" * 60)

def get_cache_key(prompt: str) -> str:
    """
    Generate a deterministic cache key from a prompt string.

    WHY MD5 not SHA256: We're not doing security — just deduplication.
    MD5 is faster and shorter. SHA256 would work too but is overkill here.

    WHY hash the prompt: Same prompt → same hash → same cached response.
    If you change the prompt even slightly, you get a new hash → new API call.
    This means you can iterate on prompts without re-calling for unchanged ones.
    """
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


# Simulating the cache pattern you'll use in generator.py
prompt_1 = "Generate a beginner plumbing repair guide"
prompt_2 = "Generate a beginner plumbing repair guide"  # Same prompt
prompt_3 = "Generate an advanced electrical repair guide"  # Different prompt

key_1 = get_cache_key(prompt_1)
key_2 = get_cache_key(prompt_2)
key_3 = get_cache_key(prompt_3)

print(f"Prompt 1 hash: {key_1}")
print(f"Prompt 2 hash: {key_2}  (same as 1? {key_1 == key_2})")
print(f"Prompt 3 hash: {key_3}  (different? {key_1 != key_3})")

# The full caching pattern:
cache_store: dict[str, str] = {}

def cached_llm_call(prompt: str) -> str:
    """
    Pattern you'll implement in generator.py:
    1. Hash the prompt
    2. Check if hash exists in cache
    3. If yes: return cached response (FREE, instant)
    4. If no: call LLM API, store in cache, return response
    """
    cache_key = get_cache_key(prompt)

    if cache_key in cache_store:
        print(f"  CACHE HIT for {cache_key[:8]}...")
        return cache_store[cache_key]

    # In real code, this is where you'd call openai.chat.completions.create()
    response = f"[Simulated LLM response for: {prompt[:30]}...]"
    cache_store[cache_key] = response
    print(f"  CACHE MISS for {cache_key[:8]}... (would call API)")
    return response


print("\n--- Caching Demo ---")
cached_llm_call("Generate beginner plumbing guide")   # MISS
cached_llm_call("Generate beginner plumbing guide")   # HIT (no API call!)
cached_llm_call("Generate advanced electrical guide")  # MISS (different prompt)
print()


# ============================================================================
# WRAP-UP: WHAT YOU NOW KNOW FOR P1 TOMORROW
# ============================================================================

print("=" * 60)
print("CHECKLIST: Ready for P1 Tomorrow?")
print("=" * 60)
print("""
✅ Type hints      → You know str, int, list[str], dict[str, int], str | None
✅ f-strings       → You can build LLM prompts with f\"\"\"...\"\"\" and escape {{ }}
✅ Comprehensions  → You can transform and filter data in one line
✅ Pydantic        → You can define models, add validators, parse LLM JSON
✅ Gotchas         → You know about mutable defaults, __name__ guard, pathlib
✅ Caching         → You know the hash-based caching pattern for LLM responses

Tomorrow you'll combine ALL of these into:
  schemas.py    → Pydantic models (Section 4)
  generator.py  → f-string prompts + caching + model_validate_json()
  test_schemas.py → pytest with valid/invalid data
""")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
