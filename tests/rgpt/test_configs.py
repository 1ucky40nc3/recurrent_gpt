from rgpt.configs import (
    LLMConfig,
    PromptConfig,
    load_llm_config,
    load_prompt_config
)


def test_llm_config():
    type = 'OpenAI'
    config = {}
    llm_config = LLMConfig(
        type=type,
        config=config
    )
    assert llm_config.type == type
    assert llm_config.config == config


def test_load_llm_config():
    path = 'tests/configs/llms/openai.json'
    llm_config = load_llm_config(path)
    assert llm_config.type is not None
    assert llm_config.config is not None


def test_prompt_config():
    input_variables = []
    template = ''
    prompt_config = PromptConfig(
        input_variables=input_variables,
        template=template
    )
    assert prompt_config.input_variables == input_variables
    assert prompt_config.template == template


def test_load_prompt_config():
    path = 'tests/configs/prompts/default.json'
    prompt_config = load_prompt_config(path)
    assert prompt_config.input_variables is not None
    assert prompt_config.template is not None