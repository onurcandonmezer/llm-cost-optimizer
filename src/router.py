"""Smart request routing engine for LLM Cost Optimizer.

Routes requests to the most cost-effective model based on task complexity,
budget constraints, and quality requirements.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from src.models import (
    Complexity,
    ModelConfig,
    QualityTier,
    RoutingDecision,
    RoutingRequest,
)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "routing_config.yaml"


class SmartRouter:
    """Routes LLM requests to optimal models based on complexity and cost.

    The router analyzes incoming text to classify its complexity, then selects
    the cheapest model from the appropriate quality tier. It supports fallback
    chains when preferred tiers are unavailable or over budget.
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """Initialize the router with model configurations.

        Args:
            config_path: Path to YAML configuration file. Uses default if None.
        """
        config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._load_config(config_path)

    def _load_config(self, config_path: Path) -> None:
        """Load routing configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.models: list[ModelConfig] = [ModelConfig(**m) for m in config["models"]]
        self.routing_rules: dict[str, str] = config["routing_rules"]
        self.fallback_chain: dict[str, list[str]] = config["fallback_chain"]
        self.complexity_thresholds: dict = config.get("complexity_thresholds", {})

        # Index models by quality tier for fast lookup
        self._models_by_tier: dict[QualityTier, list[ModelConfig]] = {}
        for model in self.models:
            tier = model.quality_tier
            if tier not in self._models_by_tier:
                self._models_by_tier[tier] = []
            self._models_by_tier[tier].append(model)

        # Sort each tier by cost (cheapest first, using input cost as proxy)
        for tier in self._models_by_tier:
            self._models_by_tier[tier].sort(key=lambda m: m.cost_per_1k_input)

    def classify_complexity(self, text: str) -> Complexity:
        """Classify the complexity of input text using heuristics.

        Uses multiple signals to determine complexity:
        - Text length
        - Presence of complexity-indicating keywords
        - Question structure and depth indicators
        - Domain-specific terminology

        Args:
            text: The input text to classify.

        Returns:
            Complexity level: simple, moderate, or complex.
        """
        text_lower = text.lower().strip()

        if not text_lower:
            return Complexity.SIMPLE

        score = 0.0

        # --- Length-based scoring ---
        char_count = len(text_lower)
        short_max = self.complexity_thresholds.get("short_text_max", 100)
        medium_max = self.complexity_thresholds.get("medium_text_max", 500)

        if char_count <= short_max:
            score += 0.0
        elif char_count <= medium_max:
            score += 1.0
        else:
            score += 2.0

        # --- Keyword-based scoring ---
        complex_keywords = self.complexity_thresholds.get("complex_keywords", [])
        simple_keywords = self.complexity_thresholds.get("simple_keywords", [])

        complex_matches = sum(1 for kw in complex_keywords if kw in text_lower)
        simple_matches = sum(1 for kw in simple_keywords if kw in text_lower)

        score += complex_matches * 1.5
        score -= simple_matches * 1.0

        # --- Question complexity ---
        question_count = text_lower.count("?")
        if question_count > 2:
            score += 1.5
        elif question_count > 0:
            score += 0.5

        # --- Structural complexity indicators ---
        # Multi-part requests
        numbered_items = len(re.findall(r"(?:^|\n)\s*\d+[\.\)]\s", text))
        bullet_items = len(re.findall(r"(?:^|\n)\s*[-*]\s", text))
        list_items = numbered_items + bullet_items
        if list_items >= 3:
            score += 2.0
        elif list_items >= 1:
            score += 0.5

        # Code-related indicators
        code_indicators = ["```", "def ", "class ", "function ", "import ", "SELECT ", "CREATE "]
        if any(ind in text for ind in code_indicators):
            score += 1.5

        # --- Word count ---
        word_count = len(text_lower.split())
        if word_count > 200:
            score += 1.5
        elif word_count > 50:
            score += 0.5

        # --- Classify based on score ---
        if score <= 1.0:
            return Complexity.SIMPLE
        elif score <= 3.5:
            return Complexity.MODERATE
        else:
            return Complexity.COMPLEX

    def _get_cheapest_model(
        self,
        tier: QualityTier,
        max_cost: Optional[float] = None,
        estimated_input_tokens: int = 500,
        estimated_output_tokens: int = 500,
    ) -> Optional[ModelConfig]:
        """Get the cheapest model in a tier that fits within budget.

        Args:
            tier: Quality tier to search.
            max_cost: Maximum acceptable cost (None for no limit).
            estimated_input_tokens: Expected input token count for cost estimation.
            estimated_output_tokens: Expected output token count for cost estimation.

        Returns:
            The cheapest suitable model, or None if no model fits.
        """
        models = self._models_by_tier.get(tier, [])

        for model in models:
            if max_cost is not None:
                estimated = model.estimate_cost(estimated_input_tokens, estimated_output_tokens)
                if estimated > max_cost:
                    continue
            return model

        return None

    def _calculate_quality_score(self, model: ModelConfig, complexity: Complexity) -> float:
        """Calculate a cost-quality optimization score.

        Higher scores indicate better cost-quality trade-offs.

        Args:
            model: The model to evaluate.
            complexity: The task complexity.

        Returns:
            Optimization score (higher is better).
        """
        tier_quality = {
            QualityTier.ECONOMY: 1.0,
            QualityTier.STANDARD: 2.0,
            QualityTier.PREMIUM: 3.0,
        }
        complexity_needs = {
            Complexity.SIMPLE: 1.0,
            Complexity.MODERATE: 2.0,
            Complexity.COMPLEX: 3.0,
        }

        quality = tier_quality[model.quality_tier]
        need = complexity_needs[complexity]

        # Penalize over-provisioning (using premium for simple tasks)
        quality_match = 1.0 - abs(quality - need) / 3.0

        # Cost efficiency (inverse of cost, normalized)
        avg_cost = model.estimate_cost(500, 500)
        cost_efficiency = 1.0 / (1.0 + avg_cost * 100)

        return quality_match * 0.6 + cost_efficiency * 0.4

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Route a request to the optimal model.

        Determines the best model by:
        1. Classifying complexity (if not pre-specified)
        2. Mapping complexity to a quality tier
        3. Finding the cheapest model in that tier
        4. Falling back to lower tiers if needed

        Args:
            request: The routing request to process.

        Returns:
            A RoutingDecision with the selected model and reasoning.

        Raises:
            ValueError: If no suitable model can be found.
        """
        # Step 1: Determine complexity
        complexity = request.complexity or self.classify_complexity(request.content)

        # Step 2: Determine target tier
        if request.required_quality:
            target_tier = request.required_quality
        else:
            tier_name = self.routing_rules.get(complexity.value, "standard")
            target_tier = QualityTier(tier_name)

        # Estimate tokens from content
        estimated_input_tokens = max(len(request.content.split()) * 2, 100)
        estimated_output_tokens = estimated_input_tokens  # rough estimate

        # Step 3: Find model in target tier
        model = self._get_cheapest_model(
            target_tier, request.max_cost, estimated_input_tokens, estimated_output_tokens
        )

        reason_parts = []
        if model:
            reason_parts.append(
                f"Complexity '{complexity.value}' mapped to '{target_tier.value}' tier"
            )
            reason_parts.append(f"Selected cheapest model in tier: {model.name}")
        else:
            # Step 4: Try fallback chain
            reason_parts.append(
                f"No suitable model in '{target_tier.value}' tier, trying fallbacks"
            )
            fallbacks = self.fallback_chain.get(target_tier.value, [])
            for fallback_tier_name in fallbacks:
                fallback_tier = QualityTier(fallback_tier_name)
                model = self._get_cheapest_model(
                    fallback_tier,
                    request.max_cost,
                    estimated_input_tokens,
                    estimated_output_tokens,
                )
                if model:
                    reason_parts.append(f"Fell back to '{fallback_tier.value}' tier: {model.name}")
                    break

        if model is None:
            raise ValueError(
                f"No suitable model found for complexity='{complexity.value}', "
                f"max_cost={request.max_cost}"
            )

        estimated_cost = model.estimate_cost(estimated_input_tokens, estimated_output_tokens)

        return RoutingDecision(
            selected_model=model,
            reason=". ".join(reason_parts),
            estimated_cost=estimated_cost,
            quality_tier=model.quality_tier,
            complexity=complexity,
        )

    def route_text(self, text: str, **kwargs) -> RoutingDecision:
        """Convenience method to route a plain text string.

        Args:
            text: The text content to route.
            **kwargs: Additional arguments passed to RoutingRequest.

        Returns:
            A RoutingDecision with the selected model and reasoning.
        """
        request = RoutingRequest(content=text, **kwargs)
        return self.route(request)

    def get_available_models(self, tier: Optional[QualityTier] = None) -> list[ModelConfig]:
        """Get available models, optionally filtered by tier.

        Args:
            tier: Optional quality tier filter.

        Returns:
            List of available model configurations.
        """
        if tier:
            return list(self._models_by_tier.get(tier, []))
        return list(self.models)
