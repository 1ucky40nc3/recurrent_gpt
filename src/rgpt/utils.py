from typing import (
    Union, 
    Iterable,
    Optional,
    Generator
) 

import string
import logging


logger = logging.getLogger(__name__)


def find_substring_between(string: str, prefix: str, suffix: str) -> Optional[str]:
    '''Find a substring between a prefix and suffix.

    Args:
        string: The string containing the substring, `prefix` and `suffix`.
        prefix: A prefix string before the substring.
        suffix: A suffix string after the substring.

    Returns:
        The substring between `prefix` and `suffix` if possible.
        We return `None` if we can't any of the args.
    '''
    try:
        start = string.index(prefix) + len(prefix)
        end = string.index(suffix, start)
        return string[start:end]
    except ValueError:
        logger.error(f"Can't return substring between '{prefix}' and '{suffix}' in '{string}'")
        return None


def find_substring_after(string: str, prefix: str) -> Optional[str]:
    '''Find a substring after a prefix.

    Args:
        string: The string containing the substring and `prefix`.
        prefix: A prefix string before the substring.

    Returns:
        The substring after `prefix` if possible.
        We return `None` if we can't any of the args.
    '''
    try:
        start = string.index(prefix) + len(prefix)
        return string[start:]
    except ValueError:
        logger.error(f"Can't return substring after '{prefix}' in '{string}'")
        return None
    

def join_paragraphs(paragraphs: Iterable[Union[None, str]], link: str = '\n') -> str:
    '''Join a number of paragraphs.
    
    Args:
        paragraphs: A iterable of paragraphs.
        link: The string we want to link between strings.

    Returns:
        The texts joined with the `link`.
    '''
    texts = map(lambda x: x or '', texts)
    return link.join(texts)


def find_format_placeholders(text: str) -> Generator[str, None, None]:
    '''Find placeholders in format strings.

    Args:
        string: A string with placeholders for formatting.
    
    Returns:
        A generator that returns the placeholders.
    '''
    for _, name, _, _ in string.Formatter().parse(text):
        yield name


def safe_format(text: str, **kwargs) -> str:
    '''Apply safe string formatting.
    
    Args:
        text: The string with placeholders we want to format.
        kwargs: Some of the placeholders with values in kwargs format.

    Returns:
        The formatted string with all placeholder-value pairs.
    '''
    placeholders = {
        placeholder: '' 
        for placeholder in find_format_placeholders(text)
    }
    return text.format(**{**placeholders, **kwargs})


def format_instructions(
    instructions: Iterable[Union[None, str]], 
    format: str = '{index}. {instruction}', 
    link: str = '\n'
) -> str:
    '''Join a number of instructions.
    
    Args:
        instructions: A iterable of paragraphs.
        format: A format string with the '{index}' and '{instruction}' fields.
        link: The string we want to link between strings.

    Returns:
        The texts joined with the `link`.

    Note:
        The fields '{index}' and '{instruction}' 
        comes from enumerating over instructions. 
    '''
    instructions = map(lambda x: x or '', instructions)
    formatted = []
    for index, instruction in enumerate(instructions):
        instruction = safe_format(format, index=index, instruction=instruction)
        formatted.append(instruction)
    return link.join(formatted)