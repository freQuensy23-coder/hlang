from typing import Optional, Iterable, Literal, Union, List, Mapping

import httpx
from openai import NotGiven, NOT_GIVEN
from openai.types.chat import ChatCompletionToolParam, ChatCompletionToolChoiceOptionParam, \
    ChatCompletionStreamOptionsParam, completion_create_params

from hlang.dataclasses.message import ChatMessage
from hlang.generators.generator import ABCChatGenerator
from hlang.openai_lazy_client import OpenAILazyClient


class OpenAIChatGenerator(ABCChatGenerator):
    def __init__(self, model_name: str, base_url: str):
        self.client = OpenAILazyClient(base_url=base_url)
        self.model_name = model_name

    def generate(self,
                 messages: [ChatMessage],
                 frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
                 function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
                 functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
                 logit_bias: Optional[dict[str, int]] | NotGiven = NOT_GIVEN,
                 logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
                 max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
                 n: Optional[int] | NotGiven = NOT_GIVEN,
                 presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
                 seed: Optional[int] | NotGiven = NOT_GIVEN,
                 service_tier: Optional[Literal["auto", "default"]] | NotGiven = NOT_GIVEN,
                 stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
                 temperature: Optional[float] | NotGiven = NOT_GIVEN,
                 top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
                 top_p: Optional[float] | NotGiven = NOT_GIVEN,
                 user: str | NotGiven = NOT_GIVEN,

                 extra_headers=None,
                 extra_query: Mapping[str, object] | None = None,
                 extra_body=None,
                 timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
                 ) -> ChatMessage | Iterable[ChatMessage]:
        self._prompt_sanity_check(messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[m.to_dict() for m in messages],
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            stream=False,
            temperature=temperature,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout
        )
        contents = [choice.message.content for choice in response.choices]
        if not n or n == 1:
            return ChatMessage.from_assistant(contents[0])
        return [ChatMessage.from_assistant(content) for content in contents]

    def stream(self,
               messages: [ChatMessage],
               frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
               function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
               functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN,
               logit_bias: Optional[dict[str, int]] | NotGiven = NOT_GIVEN,
               logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
               max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
               presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
               response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
               seed: Optional[int] | NotGiven = NOT_GIVEN,
               service_tier: Optional[Literal["auto", "default"]] | NotGiven = NOT_GIVEN,
               stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
               temperature: Optional[float] | NotGiven = NOT_GIVEN,
               top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
               top_p: Optional[float] | NotGiven = NOT_GIVEN,
               user: str | NotGiven = NOT_GIVEN,
               stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NotGiven,
               extra_headers=None,
               extra_query: Mapping[str, object] | None = None,
               extra_body=None,
               timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
               ) -> Iterable[str]:
        iterator = self.client.chat.completions.create(
            messages=[m.to_dict() for m in messages],
            model=self.model_name,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            stream=True,
            stream_options=stream_options,
            temperature=temperature,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout
        )
        for chunk in iterator:
            yield chunk.choices[0].delta.content
