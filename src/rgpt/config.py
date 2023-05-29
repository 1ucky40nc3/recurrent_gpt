'''Copyright 2023 Louis Wendler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from typing import (
    Any,
    Dict,
    List
)

import json
from dataclasses import dataclass

import langchain


def load_json(path: str) -> Dict[str, Any]:
    '''Load JSON data as a dict.
    
    Args:
        path: The path of the JSON file.
    
    Returns:
        The JSON data as a dictionary.
    '''
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@dataclass
class LLMConfig:
    '''Large Language Model (LLM) Configuration.
    
    Attributes:
        type: The type of langchain LLM integration we want to use.
        config: The config we want to instiate the LLM with.
    '''
    type: str
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        # Check if the type is a langchain integration
        if type not in dir(langchain.llms):
            raise ValueError(f"Can't a langchain implementation of the LLM type `{self.type}`!")


@dataclass
class PromptConfig:
    '''Prompt config to build langchain PromptTemplates and more.
    
    Attributes: 
        input_variables: A list of input placeholder variables in the prompt.
        partial_variables: A list of partial placeholder values in the prompt.
        template: A prompt template.
    '''
    input_variables: List[str]
    partial_variables: List[str]
    template: str


def load_llm_config(path: str) -> LLMConfig:
    '''Load some LLM config.
    
    Args:
        path: The path of the config JSON file.

    Returns:
        The LLM config from the config file.
    '''
    config = load_json(path)
    return LLMConfig(**config)


def load_prompt_config(path: str) -> PromptConfig:
    '''Load some prompt config.
    
    Args:
        path: The path of the config JSON file.

    Returns:
        The prompt config from the config file.
    '''
    config = load_json(path)
    return PromptConfig(**config)