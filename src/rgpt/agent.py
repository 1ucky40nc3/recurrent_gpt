'''Copyright 2023 Louis Wendler

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from collections import OrderedDict

from rgpt.chain import (
    load_llm, 
    create_llm_chain
)
from rgpt.vectorstore import create_vectorstore
from rgpt.configs import (
    load_llm_config,
    load_embeddings_config
)
from rgpt.utils import (
    find_substring_between,
    find_substring_after
)


class RecurrentGPTAgent:
    def __init__(
        self,
        llm_config_path: str,
        embeddings_config_path: str,
        prompt_config_path: str,
        init_prompt_config_path: str,
    ) -> str:
        self.llm_config_path = llm_config_path
        self.embeddings_config_path = embeddings_config_path
        self.prompt_config_path = prompt_config_path
        self.init_prompt_config_path = init_prompt_config_path
        
        self.llm_config = load_llm_config(self.llm_config_path)
        self.embeddings_config = load_embeddings_config(self.embeddings_config_path)
        self.llm = load_llm(self.llm_config)
        self.llm_chain = create_llm_chain(self.prompt_config_path, llm=self.llm)
        self.vectorstore = create_vectorstore(self.embeddings_config)

        self.init()

    def init(self) -> None:
        llm_chain = create_llm_chain(self.init_prompt_config_path, llm=self.llm)
        outputs = llm_chain({})
        text = outputs['text']

        paragraphs = OrderedDict([
            ('Name', None),
            ('Outline', None),
            ('Paragraph 1', None),
            ('Paragraph 2', None),
            ('Paragraph 3', None),
            ('Summary', None),
            ('Instruction 1', None),
            ('Instruction 2', None),
            ('Instruction 3', None)
        ])
        # Populate the paragraphs
        iter0 = list(paragraphs.keys())
        iter1 = list(paragraphs.keys())[1:]
        for key0, key1 in zip(iter0, iter1):
            prefix = f'{key0}:'
            suffix = f'\n{key1}:'
            paragraphs[key0] = find_substring_between(text, prefix, suffix)
        # Set the last paragraph individually 
        prefix = f'{iter0[-1]}:'
        paragraphs[iter0[-1]] = find_substring_after(text, prefix)

        self.paragraphs = paragraphs