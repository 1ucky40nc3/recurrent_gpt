from rgpt.configs import (
    LLMConfig,
    PromptConfig,
    EmbeddingsConfig,
    load_llm_config,
    load_prompt_config,
    load_embeddings_config
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
    partial = {}
    template_file_path = ''
    prompt_config = PromptConfig(
        input_variables=input_variables,
        partial=partial,
        template_file_path=template_file_path
    )
    assert prompt_config.input_variables == input_variables
    assert prompt_config.partial == partial
    assert prompt_config.template_file_path == template_file_path


def test_load_prompt_config():
    path = 'tests/configs/prompts/default.json'
    prompt_config = load_prompt_config(path)
    assert prompt_config.input_variables is not None
    assert prompt_config.partial is not None
    assert prompt_config.template_file_path is not None


def test_embeddings_config():
    type = 'HuggingFaceEmbeddings'
    config = {}
    embeddings_config = EmbeddingsConfig(
        type=type,
        config=config
    )
    assert embeddings_config.type == type
    assert embeddings_config.config == config


def test_load_embeddings_config():
    path = 'tests/configs/embeddings/huggingface.json'
    embeddings_config = load_embeddings_config(path)
    assert embeddings_config.type is not None
    assert embeddings_config.config is not None