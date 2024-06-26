from typing import Literal

Mode = Literal[
    "human",
    "augmented",
    "augmented-cached",
    "machine-openai",
    "machine-anthropic",
    "fake",
    "test",
    "approval",
    "machine-cached",
]
