from typing import Any

from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.llms import Ollama

from ..config import settings
from ..utils.logger import logger


class LLMFactory:
    """Factory class for creating and managing different LLM instances."""

    def __init__(self):
        """Initialize the LLM factory with supported models."""
        self.supported_models = {
            "gpt-4": self._create_gpt4,
            "gpt-3.5-turbo": self._create_gpt35,
            "claude-3-sonnet": self._create_claude,
            "llama2": self._create_llama2,
        }

    def create_llm(self, model_name: str) -> Any:
        """
        Create and return an LLM instance based on the model name.

        Args:
            model_name: Name of the model to create

        Returns:
            An instance of the requested LLM
        """
        try:
            if model_name not in self.supported_models:
                models = list(self.supported_models.keys())
                msg = (
                    f"Model {model_name} not supported. " f"Available models: {models}"
                )
                raise ValueError(msg)

            return self.supported_models[model_name]()

        except Exception as e:
            logger.error(f"Error creating LLM instance: {e}")
            raise

    def _create_gpt4(self) -> ChatOpenAI:
        """Create a GPT-4 instance."""
        return ChatOpenAI(
            model_name="gpt-4", temperature=0.7, api_key=settings.openai_api_key
        )

    def _create_gpt35(self) -> ChatOpenAI:
        """Create a GPT-3.5 Turbo instance."""
        return ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.7, api_key=settings.openai_api_key
        )

    def _create_claude(self) -> ChatAnthropic:
        """Create a Claude instance."""
        return ChatAnthropic(
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
            api_key=settings.anthropic_api_key,
        )

    def _create_llama2(self) -> Ollama:
        """Create a Llama 2 instance using Ollama."""
        return Ollama(model="llama2", temperature=0.7)
