#!/usr/bin/env python3
"""
Generate synthetic dating compatibility pairs for training and evaluation.
Creates diverse persona-based statements with compatibility labels.
"""

import json
import random
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import time
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

class Gender(str, Enum):
    BOY = "boy"
    GIRL = "girl"

class CompatibilityLabel(int, Enum):
    INCOMPATIBLE = 0
    COMPATIBLE = 1

class DatingPair(BaseModel):
    """Pydantic model for a dating compatibility pair."""
    text_1: str = Field(..., description="First person's statement")
    text_2: str = Field(..., description="Second person's statement")
    label: CompatibilityLabel = Field(..., description="Compatibility label (0=incompatible, 1=compatible)")
    category: Optional[str] = Field(None, description="Category of the preference (lifestyle, interests, etc.)")
    subcategory: Optional[str] = Field(None, description="Subcategory of the preference")
    pair_type: Optional[str] = Field(None, description="Type of pair (compatible, incompatible, dealbreaker, subtle)")

    @field_validator('text_1', 'text_2')
    @classmethod
    def validate_text_format(cls, v):
        """Ensure text follows the format 'gender: statement'."""
        if ':' not in v:
            raise ValueError("Text must follow format 'gender: statement'")
        gender, statement = v.split(':', 1)
        if gender.strip() not in ['boy', 'girl']:
            raise ValueError("Gender must be 'boy' or 'girl'")
        if not statement.strip():
            raise ValueError("Statement cannot be empty")
        return v

    class Config:
        use_enum_values = True

class PreferenceWeight(BaseModel):
    """Model for preference importance weights."""
    preference: str
    category: str
    importance_score: int  # 1-10 scale
    reasoning: str

class CompatibilityJudgment(BaseModel):
    """Model for LLM compatibility judgment."""
    person1_preferences: List[PreferenceWeight]
    person2_preferences: List[PreferenceWeight]
    compatibility_score: int  # 0-10 scale
    is_compatible: bool  # True if score >= 6
    reasoning: str
    key_factors: List[str]

class DatasetMetadata(BaseModel):
    """Metadata for the generated dataset."""
    total_pairs: int
    compatible_pairs: int
    incompatible_pairs: int
    dealbreaker_pairs: int
    complex_pairs: int
    llm_judged_pairs: int
    realistic_pairs: int
    subtle_mismatch_pairs: int
    categories_used: List[str]
    generation_timestamp: str
    random_seed: int = 42

class LLMJudge:
    """LLM-as-a-Judge system for evaluating dating compatibility."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the LLM judge."""
        self.model_name = model_name
        self.simulate_llm = True  # Set to False when using real API

    def extract_preferences(self, text: str) -> List[str]:
        """Extract preferences from text."""
        clean_text = text.split(': ', 1)[1] if ': ' in text else text

        # Split on connectors and extract preferences
        connectors = [' but ', ' and ', ' however ', ' although ']
        parts = [clean_text]

        for connector in connectors:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(connector))
            parts = new_parts

        preferences = []
        for part in parts:
            part = part.strip()
            if part:
                preferences.append(part)

        return preferences

    def categorize_preference(self, preference: str) -> str:
        """Categorize a preference by importance."""
        pref_lower = preference.lower()

        # Core values (highest importance)
        if any(word in pref_lower for word in ['family', 'spiritual', 'religious', 'kids', 'children', 'marriage', 'values', 'honest', 'loyal']):
            return 'values'

        # Dealbreakers
        if any(word in pref_lower for word in ['smoking', 'drugs', 'drinking', 'dishonest', 'cheating', 'drama', 'controlling']):
            return 'dealbreakers'

        # Lifestyle
        if any(word in pref_lower for word in ['social', 'quiet', 'active', 'party', 'home', 'travel', 'work', 'career']):
            return 'lifestyle'

        # Interests (lowest importance)
        return 'interests'

    def get_importance_score(self, preference: str) -> int:
        """Get importance score for a preference."""
        category = self.categorize_preference(preference)

        if category == 'values':
            return random.randint(8, 10)
        elif category == 'dealbreakers':
            return 10
        elif category == 'lifestyle':
            return random.randint(5, 7)
        else:  # interests
            return random.randint(2, 5)

    def preferences_align(self, pref1: str, pref2: str) -> bool:
        """Check if two preferences align."""
        pref1_lower = pref1.lower()
        pref2_lower = pref2.lower()

        positive_words = ['love', 'enjoy', 'passionate', 'adore', 'into', 'value', 'am a']
        negative_words = ['hate', 'dislike', 'avoid', 'not into', "can't stand", 'would never']

        p1_positive = any(word in pref1_lower for word in positive_words)
        p1_negative = any(word in pref1_lower for word in negative_words)
        p2_positive = any(word in pref2_lower for word in positive_words)
        p2_negative = any(word in pref2_lower for word in negative_words)

        # Enhanced concept matching
        key_concepts = [
            'family', 'family person', 'family tradition', 'family time',
            'honest', 'honesty', 'dishonesty', 'peaceful', 'drama',
            'travel', 'music', 'sports', 'running', 'reading', 'cooking',
            'nature', 'art', 'smoking', 'party', 'quiet'
        ]

        shared_concepts = []
        for concept in key_concepts:
            if concept in pref1_lower and concept in pref2_lower:
                shared_concepts.append(concept)

        # Special handling for family-related concepts
        family_concepts = ['family', 'family person', 'family tradition', 'family time']
        family_match = any(concept in pref1_lower and concept in pref2_lower for concept in family_concepts)

        if family_match and ((p1_positive and p2_positive) or (p1_negative and p2_negative)):
            return True

        # Align if same sentiment about shared concepts
        if shared_concepts and ((p1_positive and p2_positive) or (p1_negative and p2_negative)):
            return True

        return False

    def preferences_conflict(self, pref1: str, pref2: str) -> bool:
        """Check if two preferences conflict."""
        pref1_lower = pref1.lower()
        pref2_lower = pref2.lower()

        positive_words = ['love', 'enjoy', 'passionate', 'adore', 'into', 'value', 'am a']
        negative_words = ['hate', 'dislike', 'avoid', 'not into', "can't stand", 'would never']

        p1_positive = any(word in pref1_lower for word in positive_words)
        p1_negative = any(word in pref1_lower for word in negative_words)
        p2_positive = any(word in pref2_lower for word in positive_words)
        p2_negative = any(word in pref2_lower for word in negative_words)

        # Enhanced concept matching for conflicts
        key_concepts = [
            'running', 'coffee', 'smoking', 'party', 'partying', 'quiet',
            'travel', 'music', 'sports', 'reading', 'cooking', 'nature', 'art'
        ]

        # Special dealbreaker detection
        dealbreaker_patterns = [
            ('smoking', 'would never date a smoker'),
            ('smoking', 'smoker'),
        ]

        for pattern1, pattern2 in dealbreaker_patterns:
            if pattern1 in pref1_lower and pattern2 in pref2_lower:
                if (p1_positive and p2_negative) or (p1_negative and p2_positive):
                    return True

        # Check for shared concepts with opposite sentiments
        shared_concepts = [concept for concept in key_concepts if concept in pref1_lower and concept in pref2_lower]

        if shared_concepts and ((p1_positive and p2_negative) or (p1_negative and p2_positive)):
            return True

        return False

    def simulate_compatibility_analysis(self, p1_prefs: List[str], p2_prefs: List[str]) -> Tuple[int, bool, str, List[str]]:
        """Simulate intelligent compatibility analysis."""

        # Analyze preference alignment
        high_importance_matches = 0
        high_importance_conflicts = 0
        dealbreaker_conflicts = 0

        shared_values = []
        conflicts = []

        for p1_pref in p1_prefs:
            p1_category = self.categorize_preference(p1_pref)
            p1_importance = self.get_importance_score(p1_pref)

            for p2_pref in p2_prefs:
                p2_category = self.categorize_preference(p2_pref)
                p2_importance = self.get_importance_score(p2_pref)

                # Check for semantic similarity (simplified)
                if self.preferences_align(p1_pref, p2_pref):
                    if p1_importance >= 7 or p2_importance >= 7:
                        high_importance_matches += 1
                        shared_values.append(f"Both value {p1_pref.lower()}")

                elif self.preferences_conflict(p1_pref, p2_pref):
                    if p1_category == 'dealbreakers' or p2_category == 'dealbreakers':
                        dealbreaker_conflicts += 1
                        conflicts.append(f"Dealbreaker conflict: {p1_pref} vs {p2_pref}")
                    elif p1_importance >= 7 or p2_importance >= 7:
                        high_importance_conflicts += 1
                        conflicts.append(f"Major conflict: {p1_pref} vs {p2_pref}")

        # Determine compatibility
        if dealbreaker_conflicts > 0:
            score = random.randint(1, 3)
            compatible = False
            reasoning = f"Incompatible due to dealbreaker conflicts: {', '.join(conflicts[:2])}"
            key_factors = ["dealbreaker_conflicts"]

        elif high_importance_matches >= 2 and high_importance_conflicts == 0:
            score = random.randint(7, 9)
            compatible = True
            reasoning = f"Highly compatible due to shared core values: {', '.join(shared_values[:2])}"
            key_factors = ["shared_core_values"]

        elif high_importance_matches >= 1 and high_importance_conflicts <= 1:
            score = random.randint(6, 7)
            compatible = True
            reasoning = f"Compatible with some differences. Shared values outweigh minor conflicts."
            key_factors = ["balanced_compatibility"]

        else:
            score = random.randint(3, 5)
            compatible = False
            reasoning = f"Incompatible due to conflicting core preferences: {', '.join(conflicts[:2])}"
            key_factors = ["core_conflicts"]

        return score, compatible, reasoning, key_factors

    def judge_compatibility(self, person1_text: str, person2_text: str) -> CompatibilityJudgment:
        """Judge compatibility between two people using LLM simulation."""
        # Extract key preferences from texts
        p1_prefs = self.extract_preferences(person1_text)
        p2_prefs = self.extract_preferences(person2_text)

        # Simulate intelligent weighting
        compatibility_score, is_compatible, reasoning, key_factors = self.simulate_compatibility_analysis(p1_prefs, p2_prefs)

        response_data = {
            "person1_preferences": [
                {"preference": pref, "category": self.categorize_preference(pref),
                 "importance_score": self.get_importance_score(pref),
                 "reasoning": f"Importance based on relationship impact"}
                for pref in p1_prefs
            ],
            "person2_preferences": [
                {"preference": pref, "category": self.categorize_preference(pref),
                 "importance_score": self.get_importance_score(pref),
                 "reasoning": f"Importance based on relationship impact"}
                for pref in p2_prefs
            ],
            "compatibility_score": compatibility_score,
            "is_compatible": is_compatible,
            "reasoning": reasoning,
            "key_factors": key_factors
        }

        return CompatibilityJudgment(**response_data)

class DatingPairGenerator:
    def __init__(self, use_llm_judge: bool = True):
        """Initialize the dating pair generator with optional LLM judge."""
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            self.llm_judge = LLMJudge()

        # Define categories of preferences and traits
        self.categories = {
            'lifestyle': {
                'introverted': ['quiet weekends', 'staying home', 'small gatherings', 'reading books', 'peaceful evenings'],
                'extroverted': ['parties', 'social events', 'meeting new people', 'nightlife', 'big crowds'],
                'active': ['hiking', 'gym workouts', 'sports', 'running', 'outdoor adventures'],
                'relaxed': ['Netflix marathons', 'lazy Sundays', 'sleeping in', 'casual walks', 'meditation']
            },
            'interests': {
                'music': ['classical music', 'rock concerts', 'jazz', 'electronic music', 'live music'],
                'arts': ['museums', 'art galleries', 'theater', 'painting', 'photography'],
                'food': ['cooking', 'fine dining', 'street food', 'vegetarian food', 'trying new cuisines'],
                'travel': ['backpacking', 'luxury resorts', 'city breaks', 'nature trips', 'cultural tours']
            },
            'values': {
                'family': ['family time', 'having kids', 'close family bonds', 'family traditions', 'extended family'],
                'career': ['career ambition', 'work-life balance', 'entrepreneurship', 'job stability', 'professional growth'],
                'spirituality': ['meditation', 'religious practices', 'spiritual growth', 'mindfulness', 'philosophy'],
                'environment': ['sustainability', 'eco-friendly living', 'climate action', 'organic food', 'green energy']
            },
            'dealbreakers': {
                'substances': ['smoking', 'heavy drinking', 'drug use', 'party lifestyle', 'bar scenes'],
                'personality': ['drama', 'negativity', 'jealousy', 'controlling behavior', 'dishonesty'],
                'lifestyle': ['messiness', 'laziness', 'workaholism', 'social media obsession', 'gaming addiction']
            }
        }
        
        # Positive and negative expressions
        self.positive_expressions = [
            "I love", "I'm passionate about", "I really enjoy", "I'm into", "I adore",
            "I'm all about", "I can't get enough of", "I'm obsessed with", "I'm a huge fan of"
        ]
        
        self.negative_expressions = [
            "I hate", "I can't stand", "I despise", "I'm not into", "I dislike",
            "I would never", "I avoid", "I'm not a fan of", "I don't enjoy"
        ]
        
        self.neutral_expressions = [
            "I sometimes enjoy", "I'm okay with", "I don't mind", "I'm neutral about"
        ]

    def generate_statement(self, category: str, subcategory: str, sentiment: str, gender: Gender) -> str:
        """Generate a single dating statement."""
        items = self.categories[category][subcategory]
        item = random.choice(items)

        if sentiment == 'positive':
            expression = random.choice(self.positive_expressions)
        elif sentiment == 'negative':
            expression = random.choice(self.negative_expressions)
        else:
            expression = random.choice(self.neutral_expressions)

        return f"{gender.value}: {expression} {item}"

    def generate_compatible_pair(self) -> DatingPair:
        """Generate a compatible pair (label=1)."""
        # Avoid dealbreakers category for compatible pairs
        categories = [cat for cat in self.categories.keys() if cat != 'dealbreakers']
        category = random.choice(categories)
        subcategory = random.choice(list(self.categories[category].keys()))

        # Both have same positive sentiment about same subcategory
        if random.random() < 0.7:  # 70% same positive
            text_1 = self.generate_statement(category, subcategory, 'positive', Gender.BOY)
            text_2 = self.generate_statement(category, subcategory, 'positive', Gender.GIRL)
        else:  # 30% both negative about same thing (but not dealbreakers)
            text_1 = self.generate_statement(category, subcategory, 'negative', Gender.BOY)
            text_2 = self.generate_statement(category, subcategory, 'negative', Gender.GIRL)

        return DatingPair(
            text_1=text_1,
            text_2=text_2,
            label=CompatibilityLabel.COMPATIBLE,
            category=category,
            subcategory=subcategory,
            pair_type="compatible"
        )

    def generate_incompatible_pair(self) -> DatingPair:
        """Generate an incompatible pair (label=0)."""
        # Avoid dealbreakers category for regular incompatible pairs
        categories = [cat for cat in self.categories.keys() if cat != 'dealbreakers']
        category = random.choice(categories)
        subcategory = random.choice(list(self.categories[category].keys()))

        # One positive, one negative about same thing
        text_1 = self.generate_statement(category, subcategory, 'positive', Gender.BOY)
        text_2 = self.generate_statement(category, subcategory, 'negative', Gender.GIRL)

        return DatingPair(
            text_1=text_1,
            text_2=text_2,
            label=CompatibilityLabel.INCOMPATIBLE,
            category=category,
            subcategory=subcategory,
            pair_type="incompatible"
        )

    def generate_dealbreaker_pair(self) -> DatingPair:
        """Generate a dealbreaker incompatible pair."""
        subcategory = random.choice(list(self.categories['dealbreakers'].keys()))

        # One person states dealbreaker, other admits to it
        dealbreaker_item = random.choice(self.categories['dealbreakers'][subcategory])

        text_1 = f"{Gender.BOY.value}: I would never date someone who is into {dealbreaker_item}"
        text_2 = f"{Gender.GIRL.value}: {random.choice(self.positive_expressions)} {dealbreaker_item}"

        return DatingPair(
            text_1=text_1,
            text_2=text_2,
            label=CompatibilityLabel.INCOMPATIBLE,
            category="dealbreakers",
            subcategory=subcategory,
            pair_type="dealbreaker"
        )

    def generate_complex_preference_pair(self) -> DatingPair:
        """Generate complex multi-preference pairs with 'but' and 'and' connectors."""
        # Choose two different categories for complexity
        categories = list(self.categories.keys())
        if 'dealbreakers' in categories:
            categories.remove('dealbreakers')  # Keep dealbreakers separate

        cat1, cat2 = np.random.choice(categories, 2, replace=False)
        subcat1 = random.choice(list(self.categories[cat1].keys()))
        subcat2 = random.choice(list(self.categories[cat2].keys()))

        item1 = random.choice(self.categories[cat1][subcat1])
        item2 = random.choice(self.categories[cat2][subcat2])

        # Create complex statements with connectors
        connectors = ['but', 'and', 'however', 'although']
        connector = random.choice(connectors)

        # Determine compatibility based on shared positive preference
        if random.random() < 0.6:  # 60% compatible - shared strong preference
            # Both have same strong preference for item2, different on item1
            text_1 = f"boy: I {random.choice(self.negative_expressions).lower()} {item1} {connector} I {random.choice(self.positive_expressions).lower()} {item2}"
            text_2 = f"girl: I {random.choice(self.positive_expressions).lower()} {item1} and I {random.choice(self.positive_expressions).lower()} {item2}"
            label = CompatibilityLabel.COMPATIBLE
            pair_type = "complex_compatible"
        else:  # 40% incompatible - conflicting on major preference
            # Conflict on both preferences
            text_1 = f"boy: I {random.choice(self.positive_expressions).lower()} {item1} {connector} I {random.choice(self.negative_expressions).lower()} {item2}"
            text_2 = f"girl: I {random.choice(self.negative_expressions).lower()} {item1} and I {random.choice(self.positive_expressions).lower()} {item2}"
            label = CompatibilityLabel.INCOMPATIBLE
            pair_type = "complex_incompatible"

        return DatingPair(
            text_1=text_1,
            text_2=text_2,
            label=label,
            category=f"{cat1}_and_{cat2}",
            subcategory=f"{subcat1}_vs_{subcat2}",
            pair_type=pair_type
        )

    def generate_subtle_mismatch(self) -> DatingPair:
        """Generate subtle incompatibility (different subcategories, same category)."""
        category = random.choice(['lifestyle', 'interests'])
        subcategories = list(self.categories[category].keys())

        # Pick two different subcategories that might conflict
        sub1, sub2 = random.sample(subcategories, 2)

        text_1 = self.generate_statement(category, sub1, 'positive', Gender.BOY)
        text_2 = self.generate_statement(category, sub2, 'positive', Gender.GIRL)

        # Label as incompatible if they're conflicting lifestyles
        conflicting_pairs = [
            ('introverted', 'extroverted'),
            ('active', 'relaxed')
        ]

        is_incompatible = (sub1, sub2) in conflicting_pairs or (sub2, sub1) in conflicting_pairs
        label = CompatibilityLabel.INCOMPATIBLE if is_incompatible else CompatibilityLabel.COMPATIBLE

        return DatingPair(
            text_1=text_1,
            text_2=text_2,
            label=label,
            category=category,
            subcategory=f"{sub1}_vs_{sub2}",
            pair_type="subtle_mismatch"
        )

    def generate_llm_judged_complex_pair(self) -> DatingPair:
        """Generate complex multi-preference pairs using LLM judge for labeling."""
        if not self.use_llm_judge:
            return self.generate_complex_preference_pair()

        # Choose categories and items
        categories = [cat for cat in self.categories.keys() if cat != 'dealbreakers']

        # Create more realistic complex preferences
        preference_templates = [
            # Family + Activity conflicts
            ("I {sentiment1} {activity} and I {sentiment2} family tradition and being a family person",
             "I {sentiment3} {activity} but I {sentiment4} family and family values"),

            # Values + Lifestyle
            ("I {sentiment1} {lifestyle} but I {sentiment2} honesty and loyalty",
             "I {sentiment3} {lifestyle} and I {sentiment4} honest relationships"),

            # Multiple interests with trade-offs
            ("I {sentiment1} {interest1} and I {sentiment2} {interest2} but I {sentiment3} {dealbreaker}",
             "I {sentiment4} {interest1} but I {sentiment5} {interest2} and I would never date someone into {dealbreaker}"),

            # Career vs Family balance
            ("I {sentiment1} career ambition but I {sentiment2} work-life balance and family time",
             "I {sentiment3} professional growth and I {sentiment4} family priorities"),
        ]

        template = random.choice(preference_templates)

        # Fill in the template with realistic values
        sentiments_pos = ["love", "am passionate about", "really enjoy", "value"]
        sentiments_neg = ["hate", "can't stand", "dislike", "avoid"]

        # Sample items from categories
        activities = self.categories['lifestyle']['active'] + self.categories['interests']['music']
        lifestyles = self.categories['lifestyle']['extroverted'] + self.categories['lifestyle']['introverted']
        interests1 = self.categories['interests']['arts']
        interests2 = self.categories['interests']['food']
        dealbreakers = self.categories['dealbreakers']['substances'] + self.categories['dealbreakers']['personality']

        # Create the statements
        text_1 = template[0].format(
            sentiment1=random.choice(sentiments_pos),
            sentiment2=random.choice(sentiments_pos),
            sentiment3=random.choice(sentiments_neg),
            sentiment4=random.choice(sentiments_pos),
            sentiment5=random.choice(sentiments_pos),
            activity=random.choice(activities),
            lifestyle=random.choice(lifestyles),
            interest1=random.choice(interests1),
            interest2=random.choice(interests2),
            dealbreaker=random.choice(dealbreakers)
        )

        text_2 = template[1].format(
            sentiment1=random.choice(sentiments_pos),
            sentiment2=random.choice(sentiments_pos),
            sentiment3=random.choice(sentiments_pos if random.random() > 0.5 else sentiments_neg),
            sentiment4=random.choice(sentiments_pos),
            sentiment5=random.choice(sentiments_pos),
            activity=random.choice(activities),
            lifestyle=random.choice(lifestyles),
            interest1=random.choice(interests1),
            interest2=random.choice(interests2),
            dealbreaker=random.choice(dealbreakers)
        )

        # Add gender prefixes
        full_text_1 = f"{Gender.BOY.value}: {text_1}"
        full_text_2 = f"{Gender.GIRL.value}: {text_2}"

        # Use LLM judge to determine compatibility
        try:
            judgment = self.llm_judge.judge_compatibility(full_text_1, full_text_2)
            label = CompatibilityLabel.COMPATIBLE if judgment.is_compatible else CompatibilityLabel.INCOMPATIBLE
            pair_type = f"llm_judged_{'compatible' if judgment.is_compatible else 'incompatible'}"

            # Extract category info from judgment
            categories_involved = list(set([pref.category for pref in judgment.person1_preferences + judgment.person2_preferences]))
            category = "_and_".join(categories_involved[:2])
            subcategory = f"score_{judgment.compatibility_score}"

        except Exception as e:
            # Fallback to simple logic if LLM judge fails
            label = CompatibilityLabel.COMPATIBLE if random.random() > 0.5 else CompatibilityLabel.INCOMPATIBLE
            pair_type = "llm_fallback"
            category = "mixed"
            subcategory = "unknown"

        return DatingPair(
            text_1=full_text_1,
            text_2=full_text_2,
            label=label,
            category=category,
            subcategory=subcategory,
            pair_type=pair_type
        )

    def generate_realistic_examples(self) -> List[DatingPair]:
        """Generate specific realistic examples with LLM judge validation."""
        realistic_cases = [
            # Family values outweigh activity preferences
            {
                "text_1": "boy: I love running and I love family tradition and being a family person",
                "text_2": "girl: I hate running but I am a family person",
                "expected_compatible": True,
                "reasoning": "Family values outweigh exercise preferences"
            },

            # Shared core values about honesty
            {
                "text_1": "boy: I love coffee but I hate dishonesty and drama",
                "text_2": "girl: I hate coffee but I value honesty and peaceful relationships",
                "expected_compatible": True,
                "reasoning": "Shared core values about honesty and peace"
            },

            # Dealbreaker overrides shared interests
            {
                "text_1": "boy: I love music and art but I enjoy smoking occasionally",
                "text_2": "girl: I love music and art but I would never date a smoker",
                "expected_compatible": False,
                "reasoning": "Dealbreaker (smoking) overrides shared interests"
            },

            # Career vs family balance - compatible
            {
                "text_1": "boy: I am passionate about career growth but I value work-life balance and family time",
                "text_2": "girl: I love professional development and I prioritize family relationships",
                "expected_compatible": True,
                "reasoning": "Both value career and family balance"
            },

            # Social vs quiet lifestyle with shared values
            {
                "text_1": "boy: I love quiet weekends and reading but I value deep friendships",
                "text_2": "girl: I enjoy social gatherings and parties but I cherish meaningful relationships",
                "expected_compatible": True,
                "reasoning": "Different social styles but shared value of meaningful relationships"
            }
        ]

        pairs = []
        for case in realistic_cases:
            # Use LLM judge to validate our expectations if available
            if self.use_llm_judge:
                try:
                    judgment = self.llm_judge.judge_compatibility(case["text_1"], case["text_2"])
                    actual_compatible = judgment.is_compatible

                    # Use LLM judgment, but log if it differs from expectation
                    if actual_compatible != case["expected_compatible"]:
                        print(f"⚠️  LLM disagreement: Expected {case['expected_compatible']}, got {actual_compatible}")
                        print(f"   Reasoning: {judgment.reasoning}")

                    label = CompatibilityLabel.COMPATIBLE if actual_compatible else CompatibilityLabel.INCOMPATIBLE

                except Exception:
                    # Fallback to expected result
                    label = CompatibilityLabel.COMPATIBLE if case["expected_compatible"] else CompatibilityLabel.INCOMPATIBLE
            else:
                label = CompatibilityLabel.COMPATIBLE if case["expected_compatible"] else CompatibilityLabel.INCOMPATIBLE

            pairs.append(DatingPair(
                text_1=case["text_1"],
                text_2=case["text_2"],
                label=label,
                category="realistic_example",
                subcategory=case["reasoning"].replace(" ", "_").lower(),
                pair_type="curated_realistic"
            ))

        return pairs

    def generate_unified_dataset(self, num_pairs: int = 1000) -> Tuple[List[DatingPair], DatasetMetadata]:
        """Generate a unified dataset combining both simple and enhanced features."""
        pairs = []

        # Unified distribution that includes both simple and complex cases
        num_simple_compatible = int(num_pairs * 0.2)  # 20% simple compatible
        num_simple_incompatible = int(num_pairs * 0.15)  # 15% simple incompatible
        num_dealbreakers = int(num_pairs * 0.15)  # 15% dealbreakers
        num_complex_basic = int(num_pairs * 0.1)  # 10% basic complex (no LLM)
        num_llm_complex = int(num_pairs * 0.2) if self.use_llm_judge else 0  # 20% LLM-judged complex
        num_realistic = min(10, int(num_pairs * 0.05)) if self.use_llm_judge else 0  # 5% realistic examples
        num_subtle = num_pairs - num_simple_compatible - num_simple_incompatible - num_dealbreakers - num_complex_basic - num_llm_complex - num_realistic

        print(f"🔄 Generating {num_pairs} unified dating pairs...")
        print(f"- Simple compatible: {num_simple_compatible}")
        print(f"- Simple incompatible: {num_simple_incompatible}")
        print(f"- Dealbreakers: {num_dealbreakers}")
        print(f"- Basic complex: {num_complex_basic}")
        if self.use_llm_judge:
            print(f"- LLM-judged complex: {num_llm_complex}")
            print(f"- Realistic examples: {num_realistic}")
        print(f"- Subtle mismatches: {num_subtle}")

        # Generate simple compatible pairs
        for _ in range(num_simple_compatible):
            pairs.append(self.generate_compatible_pair())

        # Generate simple incompatible pairs
        for _ in range(num_simple_incompatible):
            pairs.append(self.generate_incompatible_pair())

        # Generate dealbreaker pairs
        for _ in range(num_dealbreakers):
            pairs.append(self.generate_dealbreaker_pair())

        # Generate basic complex pairs (without LLM judge)
        for _ in range(num_complex_basic):
            pairs.append(self.generate_complex_preference_pair())

        # Generate LLM-judged complex pairs if available
        if self.use_llm_judge and num_llm_complex > 0:
            print("🤖 Generating LLM-judged complex pairs...")
            for i in range(num_llm_complex):
                if i % 20 == 0:
                    print(f"   Progress: {i}/{num_llm_complex}")
                pairs.append(self.generate_llm_judged_complex_pair())

            # Generate realistic examples
            print("📝 Adding curated realistic examples...")
            realistic_pairs = self.generate_realistic_examples()
            pairs.extend(realistic_pairs[:num_realistic])

        # Generate subtle mismatch pairs
        for _ in range(num_subtle):
            pairs.append(self.generate_subtle_mismatch())

        # Shuffle the dataset
        random.shuffle(pairs)

        # Create unified metadata
        metadata = DatasetMetadata(
            total_pairs=len(pairs),
            compatible_pairs=sum(1 for p in pairs if p.label == CompatibilityLabel.COMPATIBLE),
            incompatible_pairs=sum(1 for p in pairs if p.label == CompatibilityLabel.INCOMPATIBLE),
            dealbreaker_pairs=num_dealbreakers,
            complex_pairs=num_complex_basic + num_llm_complex,
            llm_judged_pairs=num_llm_complex,
            realistic_pairs=num_realistic,
            subtle_mismatch_pairs=num_subtle,
            categories_used=list(self.categories.keys()) + (["realistic_example"] if num_realistic > 0 else []),
            generation_timestamp=datetime.now().isoformat(),
            random_seed=42
        )

        return pairs, metadata

    def generate_dataset(self, num_pairs: int = 1000, enhanced: bool = True) -> Tuple[List[DatingPair], DatasetMetadata]:
        """Generate a complete dataset of dating pairs with optional enhanced features."""
        pairs = []

        if enhanced and self.use_llm_judge:
            # Enhanced distribution with LLM-judged complex cases
            num_simple_compatible = int(num_pairs * 0.25)  # 25% simple compatible
            num_simple_incompatible = int(num_pairs * 0.2)  # 20% simple incompatible
            num_dealbreakers = int(num_pairs * 0.15)  # 15% dealbreakers
            num_llm_complex = int(num_pairs * 0.25)  # 25% LLM-judged complex
            num_realistic = min(10, int(num_pairs * 0.05))  # 5% realistic examples (max 10)
            num_subtle = num_pairs - num_simple_compatible - num_simple_incompatible - num_dealbreakers - num_llm_complex - num_realistic

            print(f"🧠 Generating {num_pairs} enhanced dating pairs with LLM judge...")
            print(f"- Simple compatible: {num_simple_compatible}")
            print(f"- Simple incompatible: {num_simple_incompatible}")
            print(f"- Dealbreakers: {num_dealbreakers}")
            print(f"- LLM-judged complex: {num_llm_complex}")
            print(f"- Realistic examples: {num_realistic}")
            print(f"- Subtle mismatches: {num_subtle}")

            # Generate simple compatible pairs
            for _ in range(num_simple_compatible):
                pairs.append(self.generate_compatible_pair())

            # Generate simple incompatible pairs
            for _ in range(num_simple_incompatible):
                pairs.append(self.generate_incompatible_pair())

            # Generate dealbreaker pairs
            for _ in range(num_dealbreakers):
                pairs.append(self.generate_dealbreaker_pair())

            # Generate LLM-judged complex pairs
            print("🤖 Generating LLM-judged complex pairs...")
            for i in range(num_llm_complex):
                if i % 20 == 0:
                    print(f"   Progress: {i}/{num_llm_complex}")
                pairs.append(self.generate_llm_judged_complex_pair())

            # Generate realistic examples
            print("📝 Adding curated realistic examples...")
            realistic_pairs = self.generate_realistic_examples()
            pairs.extend(realistic_pairs[:num_realistic])

            # Generate subtle mismatch pairs
            for _ in range(num_subtle):
                pairs.append(self.generate_subtle_mismatch())

            # Metadata for enhanced dataset
            metadata = DatasetMetadata(
                total_pairs=len(pairs),
                compatible_pairs=sum(1 for p in pairs if p.label == CompatibilityLabel.COMPATIBLE),
                incompatible_pairs=sum(1 for p in pairs if p.label == CompatibilityLabel.INCOMPATIBLE),
                dealbreaker_pairs=num_dealbreakers,
                complex_pairs=num_llm_complex,
                llm_judged_pairs=num_llm_complex,
                realistic_pairs=num_realistic,
                subtle_mismatch_pairs=num_subtle,
                categories_used=list(self.categories.keys()) + ["realistic_example"],
                generation_timestamp=datetime.now().isoformat(),
                random_seed=42
            )

        else:
            # Original distribution for basic dataset
            num_compatible = int(num_pairs * 0.3)  # 30% compatible
            num_incompatible = int(num_pairs * 0.25)  # 25% incompatible
            num_dealbreakers = int(num_pairs * 0.2)  # 20% dealbreakers
            num_complex = int(num_pairs * 0.15)  # 15% complex multi-preference
            num_subtle = num_pairs - num_compatible - num_incompatible - num_dealbreakers - num_complex  # 10% subtle

            print(f"📊 Generating {num_pairs} basic dating pairs...")
            print(f"- Compatible: {num_compatible}")
            print(f"- Incompatible: {num_incompatible}")
            print(f"- Dealbreakers: {num_dealbreakers}")
            print(f"- Complex multi-preference: {num_complex}")
            print(f"- Subtle mismatches: {num_subtle}")

            # Generate compatible pairs
            for _ in range(num_compatible):
                pairs.append(self.generate_compatible_pair())

            # Generate incompatible pairs
            for _ in range(num_incompatible):
                pairs.append(self.generate_incompatible_pair())

            # Generate dealbreaker pairs
            for _ in range(num_dealbreakers):
                pairs.append(self.generate_dealbreaker_pair())

            # Generate complex multi-preference pairs
            for _ in range(num_complex):
                pairs.append(self.generate_complex_preference_pair())

            # Generate subtle mismatch pairs
            for _ in range(num_subtle):
                pairs.append(self.generate_subtle_mismatch())

            # Metadata for basic dataset
            metadata = DatasetMetadata(
                total_pairs=len(pairs),
                compatible_pairs=num_compatible,
                incompatible_pairs=num_incompatible,
                dealbreaker_pairs=num_dealbreakers,
                complex_pairs=num_complex,
                llm_judged_pairs=0,
                realistic_pairs=0,
                subtle_mismatch_pairs=num_subtle,
                categories_used=list(self.categories.keys()),
                generation_timestamp=datetime.now().isoformat(),
                random_seed=42
            )

        # Shuffle the dataset
        random.shuffle(pairs)

        return pairs, metadata

def save_jsonl(data: List[DatingPair], filepath: str):
    """Save data to JSONL format."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(item.model_dump_json() + '\n')

def save_json(data: DatasetMetadata, filepath: str):
    """Save metadata to JSON format."""
    with open(filepath, 'w') as f:
        f.write(data.model_dump_json(indent=2))

def main():
    """Main function with support for both basic and enhanced dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate dating compatibility dataset')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'unified'], default='unified',
                       help='Generation mode: basic (simple pairs), enhanced (with LLM judge), or unified (combines both)')
    parser.add_argument('--train-size', type=int, default=1200,
                       help='Number of training pairs to generate')
    parser.add_argument('--eval-size', type=int, default=300,
                       help='Number of evaluation pairs to generate')

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)

    # Initialize generator
    use_llm_judge = (args.mode in ['enhanced', 'unified'])
    generator = DatingPairGenerator(use_llm_judge=use_llm_judge)

    print(f"🎯 DATING COMPATIBILITY DATASET GENERATION")
    print(f"Mode: {args.mode.upper()}")
    print(f"LLM Judge: {'Enabled' if use_llm_judge else 'Disabled'}")
    print("=" * 60)

    # Generate training dataset
    print(f"\n🚀 Generating training dataset ({args.train_size} pairs)...")

    if args.mode == 'unified':
        train_pairs, train_metadata = generator.generate_unified_dataset(args.train_size)
        train_path = data_dir / 'dating_pairs.jsonl'
        train_meta_path = data_dir / 'dating_pairs_metadata.json'
    elif args.mode == 'enhanced':
        train_pairs, train_metadata = generator.generate_dataset(args.train_size, enhanced=True)
        train_path = data_dir / 'enhanced_dating_pairs.jsonl'
        train_meta_path = data_dir / 'enhanced_dating_pairs_metadata.json'
    else:  # basic mode
        train_pairs, train_metadata = generator.generate_dataset(args.train_size, enhanced=False)
        train_path = data_dir / 'dating_pairs.jsonl'
        train_meta_path = data_dir / 'dating_pairs_metadata.json'

    save_jsonl(train_pairs, train_path)
    save_json(train_metadata, train_meta_path)
    print(f"✅ Saved {len(train_pairs)} training pairs to {train_path}")
    print(f"✅ Saved training metadata to {train_meta_path}")

    # Generate evaluation dataset
    print(f"\n🔍 Generating evaluation dataset ({args.eval_size} pairs)...")

    if args.mode == 'unified':
        eval_pairs, eval_metadata = generator.generate_unified_dataset(args.eval_size)
        eval_path = data_dir / 'eval_pairs.jsonl'
        eval_meta_path = data_dir / 'eval_pairs_metadata.json'
    elif args.mode == 'enhanced':
        eval_pairs, eval_metadata = generator.generate_dataset(args.eval_size, enhanced=True)
        eval_path = data_dir / 'enhanced_eval_pairs.jsonl'
        eval_meta_path = data_dir / 'enhanced_eval_pairs_metadata.json'
    else:  # basic mode
        eval_pairs, eval_metadata = generator.generate_dataset(args.eval_size, enhanced=False)
        eval_path = data_dir / 'eval_pairs.jsonl'
        eval_meta_path = data_dir / 'eval_pairs_metadata.json'

    save_jsonl(eval_pairs, eval_path)
    save_json(eval_metadata, eval_meta_path)
    print(f"✅ Saved {len(eval_pairs)} evaluation pairs to {eval_path}")
    print(f"✅ Saved evaluation metadata to {eval_meta_path}")

    # Print examples
    print("\n" + "="*60)
    print("🎯 SAMPLE PAIRS:")
    print("="*60)

    # Show different types of examples
    example_types = {}
    for pair in train_pairs:
        pair_type = pair.pair_type
        if pair_type not in example_types:
            example_types[pair_type] = []
        if len(example_types[pair_type]) < 2:
            example_types[pair_type].append(pair)

    for pair_type, examples in example_types.items():
        print(f"\n📝 {pair_type.replace('_', ' ').title()} Examples:")
        for i, pair in enumerate(examples, 1):
            label_text = "COMPATIBLE" if pair.label == CompatibilityLabel.COMPATIBLE else "INCOMPATIBLE"
            print(f"  {i}. {label_text}:")
            print(f"     {pair.text_1}")
            print(f"     {pair.text_2}")
            if pair.category != "realistic_example":
                print(f"     Category: {pair.category} -> {pair.subcategory}")
            else:
                print(f"     Reasoning: {pair.subcategory.replace('_', ' ').title()}")

    # Print comprehensive statistics
    print(f"\n📊 DATASET STATISTICS:")
    print("=" * 60)
    print(f"� Training Dataset ({len(train_pairs)} pairs):")
    print(f"  • Compatible: {train_metadata.compatible_pairs}")
    print(f"  • Incompatible: {train_metadata.incompatible_pairs}")
    print(f"  • Dealbreakers: {train_metadata.dealbreaker_pairs}")
    print(f"  • Complex pairs: {train_metadata.complex_pairs}")
    if hasattr(train_metadata, 'llm_judged_pairs'):
        print(f"  • LLM-judged: {train_metadata.llm_judged_pairs}")
        print(f"  • Realistic examples: {train_metadata.realistic_pairs}")
    print(f"  • Subtle mismatches: {train_metadata.subtle_mismatch_pairs}")

    print(f"\n📈 Evaluation Dataset ({len(eval_pairs)} pairs):")
    print(f"  • Compatible: {eval_metadata.compatible_pairs}")
    print(f"  • Incompatible: {eval_metadata.incompatible_pairs}")

    print(f"\n🎉 Dataset generation complete!")
    print(f"🔧 Mode: {args.mode.upper()}")
    print(f"📁 Files saved in: {data_dir}")

    if args.mode in ['enhanced', 'unified']:
        print(f"🧠 Enhanced features:")
        print(f"  • LLM-judged complex preferences")
        print(f"  • Intelligent preference weighting")
        print(f"  • Curated realistic examples")
        print(f"  • Multi-preference trade-off scenarios")

        if args.mode == 'unified':
            print(f"🔄 Unified dataset benefits:")
            print(f"  • Single file for all training/evaluation")
            print(f"  • Combines simple + complex examples")
            print(f"  • Balanced distribution of difficulty levels")
            print(f"  • Ready for fine-tuning without file switching")

if __name__ == "__main__":
    main()
