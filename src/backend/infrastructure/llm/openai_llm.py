import os
from typing import List, Optional

from langchain.schema import BaseMessage, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


class OpenAILLM(BaseChatModel):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str = os.getenv("PROXYAPI_KEY"),  # type: ignore
        base_url: str = os.getenv("OPENAI_BASE_URL"),  # type: ignore
        top_p: float = 0.8,
        max_completion_tokens: int = 512,
        seed: int | None = 42,
        callbacks=None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Initializes the OpenAILLM.

        Args:
            model_name (str): Name of the model to use. Defaults to "gpt-4o-mini".
            temperature (float): Temperature to use for generation. Defaults to 0.0.
            api_key (str): API key to use for the OpenAI API. Defaults to os.getenv("PROXYAPI_KEY").
            base_url (str): Base URL to use for the OpenAI API. Defaults to os.getenv("OPENAI_BASE_URL").
            top_p (float): Top-p value to use for generation. Defaults to 0.8.
            max_completion_tokens (int): Maximum number of tokens to generate. Defaults to 512.
            seed (int | None): Seed to use for generation. Defaults to 42.
            callbacks (None): Callbacks to use. Defaults to None.
            tags (Optional[list[str]]): Tags to use. Defaults to None.
            metadata (Optional[dict]): Metadata to use. Defaults to None.

        Returns:
            None
        """
        super().__init__(callbacks=callbacks, tags=tags, metadata=metadata)

        self._llm = ChatOpenAI(
            name=model_name,
            temperature=temperature,
            api_key=api_key,  # type: ignore
            base_url=base_url,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            seed=seed,
        )

    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """
        Generate a chat result by processing the given messages.

        Args:
            messages (List[BaseMessage]): A list of messages to generate a response for.
            stop (Optional[List[str]]): A list of stop sequences to use during generation, if any.
            run_manager: An optional manager for handling the run, defaults to None.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            ChatResult: The result of the chat generation process.
        """
        return self._llm._generate(  # pylint: disable=protected-access
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _agenerate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return await self._llm._agenerate(  # pylint: disable=protected-access
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    # (опционально) сообщаем LangChain тип модели
    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "openai-wrapper"
