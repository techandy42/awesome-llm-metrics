from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the model with an API key and model name."""
        pass

    @abstractmethod
    def __str__(self):
        """Return the name of the model."""
        pass

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Make a general call to the model with a prompt."""
        pass

    @abstractmethod
    def translate(self, prompt: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from source language to target language."""
        pass

    @abstractmethod
    def summarize(self, prompt: str) -> str:
        """Summarize the given text."""
        pass

    @abstractmethod
    def q_and_a(self, prompt: str) -> str:
        """Provide an answer to the given question."""
        pass

    @abstractmethod
    def complete_sentence(self, prompt: str) -> str:
        """Complete the given sentence."""
        pass

    @abstractmethod
    def complete_missing_word(self, prompt: str) -> str:
        """Complete the missing word in the given sentence."""
        pass
