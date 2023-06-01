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

from typing import Optional

from dataclasses import dataclass
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
    find_substring_after,
    join_paragraphs,
    join_instructions
)


@dataclass
class ActorContext:
    input_paragraph: Optional[str] = None
    output_paragraph: Optional[str] = None
    output_memory: Optional[str] = None
    output_instruction: Optional[str] = None


@dataclass
class ModelContext:
    input_paragraph: Optional[str] = None
    input_instruction: Optional[str] = None
    short_memory: Optional[str] = None
    long_memory: Optional[str] = None


class RecurrentGPTAgent:
    def __init__(
        self,
        llm_config_path: str,
        embeddings_config_path: str,
        model_prompt_config_path: str,
        init_prompt_config_path: str,
        actor_prompt_config_path: str,
        plan_prompt_config_path: str,
        num_paragraphs: int = 3,
        num_instructions: int = 3,
        name_label: str = 'Name',
        outline_label: str = 'Outline',
        paragraph_label: str = 'Paragraph',
        summary_label: str = 'Summary',
        instruction_label: str = 'Instruction',
        plan_prefix_label: str = 'Selected Plan:',
        plan_suffix_label: str = 'Reason',
        new_paragraph_prefix_label: str = 'Extended Paragraph:',
        new_paragraph_suffix_label: str = 'Selected Plan',
        new_plan_prefix_label: str = 'Revised Plan:'
    ) -> str:
        self.llm_config_path = llm_config_path
        self.embeddings_config_path = embeddings_config_path
        self.model_prompt_config_path = model_prompt_config_path
        self.init_prompt_config_path = init_prompt_config_path
        self.actor_prompt_config_path = actor_prompt_config_path
        self.plan_prompt_config_path = plan_prompt_config_path
        self.num_paragraphs = num_paragraphs
        self.num_instructions = num_instructions
        self.name_label = name_label
        self.outline_label = outline_label
        self.paragraph_label = paragraph_label
        self.summary_label = summary_label
        self.instruction_label = instruction_label
        self.plan_prefix_label = plan_prefix_label
        self.plan_suffix_label = plan_suffix_label
        self.new_paragraph_prefix_label = new_paragraph_prefix_label
        self.new_paragraph_suffix_label = new_paragraph_suffix_label
        self.new_plan_prefix_label = new_plan_prefix_label

        self.llm_config = load_llm_config(self.llm_config_path)
        self.llm = load_llm(self.llm_config)
        self.embeddings_config = load_embeddings_config(self.embeddings_config_path)
        self.vectorstore = create_vectorstore(self.embeddings_config)

        self.actor = ActorContext()
        self.model = ModelContext()

        self.init()

    def init(self) -> None:
        llm_chain = create_llm_chain(self.init_prompt_config_path, llm=self.llm)
        outputs = llm_chain({})
        text = outputs['text']

        paragraphs = OrderedDict([
            (self.name_label, None),
            (self.outline_label, None),
            *[
                (f'{self.paragraph_label} {i}', None) 
                for i in range(1, self.num_paragraphs + 1)
            ],
            (self.summary_label, None),
            *[
                (f'{self.instruction_label} {i}', None) 
                for i in range(1, self.num_instructions + 1)
            ]
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
        input_paragraphs = [f'{self.paragraph_label} {i}' for i in range(1, self.num_paragraphs)]
        input_paragraphs = [self.paragraphs[key] for key in input_paragraphs]
        self.actor.input_paragraph = join_paragraphs(input_paragraphs)
        self.actor.output_paragraph = self.paragraphs[f'{self.paragraph_label} {self.num_paragraphs}']
        self.actor.output_memory = self.paragraphs[self.summary_label]
        output_instructions = [f'{self.instruction_label} {i}' for i in range(1, self.num_instructions + 1)]
        output_instructions = [self.paragraphs[key] for key in output_instructions]
        self.actor.output_instruction = join_instructions(output_instructions)

        self.actor_step()

        self.model.input_paragraph = self.actor.output_paragraph
        self.model.input_instruction = self.actor.output_instruction
        self.model.short_memory = self.paragraphs[self.summary_label]
        self.vectorstore.add_texts(input_paragraphs)

    def plan(self) -> None:
        previous_paragraph = self.actor.input_paragraph
        writer_new_paragraph = self.actor.output_paragraph
        memory = self.actor.output_memory
        previous_plans = self.actor.output_instruction
        
        llm_chain = create_llm_chain(self.plan_prompt_config_path, llm=self.llm)
        plan = None
        while plan is None:
            response = llm_chain.run(
                previous_paragraph=previous_paragraph,
                writer_new_paragraph=writer_new_paragraph,
                memory=memory,
                previous_plans=previous_plans
            )
            plan = find_substring_between(
                response, 
                self.plan_prefix_label, 
                self.plan_suffix_label
            )
        
        self.actor.output_instruction = plan

    def actor_step(self) -> None:
        self.plan()

        previous_paragraph = self.actor.input_paragraph
        writer_new_paragraph = self.actor.output_paragraph
        memory = self.actor.output_memory
        user_edited_plan = self.actor.output_instruction

        llm_chain = create_llm_chain(self.actor_prompt_config_path, llm=self.llm)
        new_paragraph = None
        new_instruction = None
        while not (new_paragraph is not None and new_instruction is not None): 
            response = llm_chain.run(
                previous_paragraph=previous_paragraph,
                writer_new_paragraph=writer_new_paragraph,
                memory=memory,
                user_edited_plan=user_edited_plan
            )
            lines = response.split('\n')
            if self.new_paragraph_prefix_label in lines[0]:
                new_paragraph = find_substring_between(
                    response, 
                    self.new_paragraph_prefix_label, 
                    self.new_paragraph_suffix_label
                )
            else:
                new_paragraph = lines[0]
            new_instruction = find_substring_after(
                response,
                self.new_plan_prefix_label
            )

        self.model.input_paragraph = new_paragraph
        self.model.input_instruction = new_instruction
    
    def model_step(self) -> None:
        pass

    def step(self) -> None:
        self.model_step()
        self.actor_step()