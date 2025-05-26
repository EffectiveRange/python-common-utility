import time
from difflib import Differ
from pathlib import Path
from typing import Callable, Any, Optional, Union


def compare_files(file1: Union[str, Path], file2: Union[str, Path], exclude_lines: Optional[list[str]] = None) -> bool:
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    return compare_lines(lines1, lines2, exclude_lines)


def compare_file_with_lines(
        file: Union[str, Path], lines: list[str], exclude_lines: Optional[list[str]] = None) -> bool:
    with open(file, 'r') as f1:
        file_lines = f1.readlines()

    return compare_lines(file_lines, lines, exclude_lines)


def compare_lines(lines1: list[str], lines2: list[str], exclude_lines: Optional[list[str]] = None) -> bool:
    if exclude_lines is None:
        exclude_lines = []

    all_lines_match = True

    for line in Differ().compare(lines1, lines2):
        line = line.strip()
        if not any(keyword in line for keyword in exclude_lines):
            if not line.startswith('?'):
                print(line)
            if line.startswith(('-', '+')):
                all_lines_match = False
        elif line.startswith('+'):
            print(line.replace('+ ', ''))

    return all_lines_match


def wait_for_condition(timeout: float, condition: Callable[..., bool], *args: Any, **kwargs: Any) -> None:
    time_step = 0.01
    total_time = 0.0

    while total_time <= timeout:
        if args or kwargs:
            if condition(*args, **kwargs):
                return
        else:
            if condition():
                return
        total_time += time_step
        time.sleep(time_step)
    else:
        raise TimeoutError(f'Failed to meet condition in {timeout} seconds')


def wait_for_assertion(timeout: float, assertion: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    time_step = 0.01
    total_time = 0.0
    error_message = ''

    while total_time <= timeout:
        try:
            if args or kwargs:
                assertion(*args, **kwargs)
            else:
                assertion()
                print(total_time)
            return
        except AssertionError as error:
            error_message = str(error)
            total_time += time_step
            time.sleep(time_step)
    else:
        raise AssertionError(f'Failed to assert in {timeout} seconds: {error_message}')
