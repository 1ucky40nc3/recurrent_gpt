from typing import Optional

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