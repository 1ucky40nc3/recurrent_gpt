import dotenv
dotenv.load_dotenv()

from rgpt.configs import load_prompt_config

from rgpt.chain import (
    load_prompt,
    create_llm_chain
)


def test_load_prompt():
    prompt_config_path = 'tests/configs/prompts/default.json'
    prompt_config = load_prompt_config(prompt_config_path)
    prompt = load_prompt(prompt_config)
    assert prompt is not None


def test_load_prompt_init():
    prompt_config_path = 'tests/configs/prompts/init.json'
    prompt_config = load_prompt_config(prompt_config_path)
    prompt = load_prompt(prompt_config)
    assert prompt is not None
    assert prompt.format() is not None


def test_create_llm_chain():
    prompt_config_path = 'tests/configs/prompts/default.json'
    llm_config_path = 'tests/configs/llms/openai.json'

    llm_chain = create_llm_chain(prompt_config_path, llm_config_path)
    assert llm_chain is not None


def test_create_llm_chain_init():
    prompt_config_path = 'tests/configs/prompts/init.json'
    llm_config_path = 'tests/configs/llms/openai.json'

    llm_chain = create_llm_chain(prompt_config_path, llm_config_path)
    assert llm_chain is not None