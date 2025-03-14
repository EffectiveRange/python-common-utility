# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import os
import shutil
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from context_logger import get_logger

log = get_logger('ConfigLoader')


class IConfigLoader(object):

    def load(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError()


class ConfigLoader(IConfigLoader):

    def __init__(self, resource_root: str, default_config: str, config_file_argument: str = 'config_file') -> None:
        self._resource_root = resource_root
        self._default_config = f'{self._resource_root}/{default_config}'
        self._config_file_argument = config_file_argument

    def load(self, arguments: dict[str, Any]) -> dict[str, Any]:
        config_file = Path(arguments[self._config_file_argument])

        if not os.path.exists(config_file):
            log.info('Loading default configuration file', config_file=self._default_config)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._default_config, config_file)
        else:
            log.info('Using configuration file', config_file=str(config_file))

        parser = ConfigParser()
        parser.read(config_file)

        configuration = {}

        for section in parser.sections():
            configuration.update(dict(parser[section]))

        configuration.update(arguments)

        return configuration
