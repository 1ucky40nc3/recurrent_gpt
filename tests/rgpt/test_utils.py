from rgpt.utils import (
    find_substring_between,
    find_substring_after
)


def test_find_substring_between():
    prefix = 'prefix: '
    suffix = ' \nsuffix:'
    substring = 'test'
    string = f'{prefix}{substring}{suffix}'
    result = find_substring_between(string, prefix, suffix)
    assert result == substring


def test_find_substring_after():
    prefix = 'prefix: '
    substring = 'test'
    string = f'{prefix}{substring}'
    result = find_substring_after(string, prefix)
    assert result == substring