# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import os
import re
import shutil
from os.path import exists
from typing import Any

from jinja2 import Environment, FileSystemLoader


def create_directory(directory: str) -> None:
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def delete_directory(directory: str) -> None:
    if os.path.isdir(directory):
        shutil.rmtree(directory)


def create_file(file_path: str, content: str = '\n') -> None:
    create_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        f.write(content)


def copy_file(source: str, destination: str) -> None:
    create_directory(os.path.dirname(destination))
    shutil.copy(source, destination)


def append_file(file_path: str, line: str) -> None:
    with open(file_path, 'a+') as file:
        file.writelines([f'{line}\n'])


def delete_file(file_path: str) -> None:
    if exists(file_path):
        if os.path.islink(file_path):
            os.unlink(file_path)
        else:
            os.remove(file_path)


def is_file_matches_pattern(file_path: str, pattern: str) -> bool:
    if not exists(file_path):
        return False

    with open(file_path) as file:
        return re.search(pattern, file.read(), re.MULTILINE) is not None


def is_file_contains_lines(file: str, expected_lines: list[str]) -> bool:
    if not exists(file):
        return False

    with open(file) as f:
        file_lines = f.read().splitlines()

    file_lines_set = set(file_lines)
    expected_lines_set = set(expected_lines)
    return file_lines_set == expected_lines_set


def render_template_file(resource_root: str, template_file: str, context: dict[str, Any]) -> str:
    template_path = f'{resource_root}/{template_file}'
    environment = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = environment.get_template(os.path.basename(template_path))
    return f'{template.render(context)}\n'
