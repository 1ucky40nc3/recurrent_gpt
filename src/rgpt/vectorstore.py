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

import langchain
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS

from rgpt.configs import (
    EmbeddingsConfig,
    load_embeddings_config
)


def load_embeddings(embeddings_config: EmbeddingsConfig) -> Embeddings:
    '''Load a text embedding model from config.
    
    Args:
        embeddings_config: Some embeddings config.

    Returns:
        A langchain text embedding model instance.
    '''
    cls = getattr(langchain.embeddings, embeddings_config.type)
    return cls(**embeddings_config.config)


def create_vectorstore(embeddings_config: EmbeddingsConfig) -> FAISS:
    embeddings = load_embeddings(embeddings_config)
    return FAISS(embedding_function=embeddings)