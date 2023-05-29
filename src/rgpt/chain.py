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

from typing import Optional

import langchain

from rgpt import llms
from rgpt.configs import (
    LLMConfig,
    PromptConfig,
    load_llm_config,
    load_prompt_config
)


def load_prompt(prompt_config: PromptConfig) -> langchain.PromptTemplate:
    '''Load a prompt template based on config.
    
    Args:
        prompt_config: The prompt config.

    Returns:
        A langchain PromptTemplate instance.
    '''
    with open(prompt_config.template_file_path, 'r', encoding='utf-8') as f:
        template = f.read()

    prompt = langchain.PromptTemplate(
        input_variables=prompt_config.input_variables,
        template=template
    )
    prompt = prompt.partial(**prompt_config.partial)
    return prompt


def load_llm(llm_config: LLMConfig) -> llms.BaseLLM:
    '''Load a LLM based on the config.
    
    Args:
        llm_config: Some LLM config.

    Returns:
        A LLM instance.
    '''
    cls = getattr(llms, eval(llm_config.type))
    return cls(**llm_config.config)


def create_llm_chain(
    prompt_config_path: str,
    llm_config_path: Optional[str] = None,
    llm: Optional[llms.BaseLLM] = None
) -> langchain.LLMChain:
    '''Create a langchain LLMChain from given config.
    
    Args:
        prompt_config_path: The path to prompt config file.
        llm_config_path: The path to a LLM config file.
        llm: An optional langchain LLM instance.
    Returns:
        A langchain LLMChain.
    '''
    prompt_config = load_prompt_config(prompt_config_path)
    prompt = load_prompt(prompt_config)
    
    if llm_config_path is not None:
        llm_config = load_llm_config(llm_config_path)
        llm = load_llm(llm_config)
    else:
        if llm is None:
            raise ValueError('One of `llm` or `llm_config_path` needs to be specified.')
        
    return langchain.LLMChain(prompt=prompt, llm=llm)