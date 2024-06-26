import math
from typing import Any
from typing import Optional
from typing import Union
from typing import List

import numpy as np
from structlog.stdlib import get_logger

from ice.agents.base import Agent
from ice.agents.base import Stop
from ice.apis.anthropic import anthropic_chatcomplete
from ice.apis.anthropic import anthropic_complete
from ice.apis.anthropic import anthropic_embedding
from ice.environment import env
from ice.utils import longest_common_prefix, n_tokens

log = get_logger()


class AnthropicAIAgent(Agent):
    """An agent that uses the Anthropic API to generate answers and predictions."""

    def __init__(
        self,
        model: str = "claude-instant-1.2",
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    async def complete(
        self,
        *,
        prompt: str,
        stop: Optional[Stop] = None,
        verbose: bool = False,
        default: str = "",
        max_tokens: int = 512,
    ) -> str:
        """Generate an answer to a question given some context."""

        if verbose:
            self._print_markdown(prompt)
        response = await self._complete(prompt, stop=stop, max_tokens=max_tokens)
        completion = self._extract_completion(response)
        if verbose:
            self._print_markdown(completion)
        return completion

    async def predict(
        self,
        *,
        context,
        default: str = "",
        verbose: bool = False,
    ) -> dict[str, float]:
        """Generate a probability distribution over the next token given some context."""
        raise NotImplementedError(
            "Anthropic API does not support logprobs therefore classification cannot be done."
        )

    async def classify(
        self,
        *,
        prompt: str,
        choices: tuple[str, ...],
        default: Optional[str] = None,
        verbose: bool = False,
    ) -> tuple[dict[str, float], Optional[str]]:
        raise NotImplementedError(
            "Anthropic API does not support logprobs therefore classification cannot be done."
        )

    async def _complete(self, prompt, **kwargs) -> dict:
        """Send a completion request to the Anthropic API with the given prompt and parameters."""
        kwargs.update(
            {
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "n": 1,
            }
        )
        response = await anthropic_complete(prompt, **kwargs)
        if "completion" not in response:
            raise ValueError(f"No completion in response: {response}")
        return response

    def _extract_completion(self, response: dict) -> str:
        """Extract the answer text from the completion response."""
        return response["completion"].strip()

    def _extract_prediction(self, response: dict) -> dict[str, float]:
        """Extract the prediction dictionary from the completion response."""
        answer = response["choices"][0]["logprobs"]["top_logprobs"][0]
        return {k: math.exp(p) for (k, p) in answer.items()}

    def _compute_relative_probs(
        self, choices: tuple[str, ...], choice_prefix: str, prediction: dict[str, float]
    ) -> dict[str, float]:
        """Compute the relative probabilities of the choices based on the prediction."""

        def lookup_prob(choice: str):
            scores = 0.0
            for token, prob in prediction.items():
                if choice[len(choice_prefix) :].startswith(token):
                    scores += prob
            return scores

        abs_probs = {choice: lookup_prob(choice) for choice in choices}
        Z = sum(abs_probs.values())
        if Z < 0.6:
            log.warning(f"{1-Z} of unaccounted probability in classify")
            log.warning(choice_prefix)
            log.warning(str(prediction))
            log.warning(str(abs_probs))

        rel_probs = (
            {choice: prob / Z for (choice, prob) in abs_probs.items()}
            if Z != 0.0
            else abs_probs
        )
        return rel_probs

    def _print_markdown(self, obj: Any):
        """Print the text with markdown formatting."""
        env().print(obj, format_markdown=True)
