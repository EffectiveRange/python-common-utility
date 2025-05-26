# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from context_logger import get_logger

from common_utility import copy_file

log = get_logger('ConfigLoader')


class IConfigLoader(object):

    def load(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError()


class ConfigLoader(IConfigLoader):

    def __init__(self, default_config_file: Path, config_file_argument: str = 'config_file') -> None:
        self._default_config_file = default_config_file
        self._config_file_argument = config_file_argument

    def load(self, arguments: dict[str, Any]) -> dict[str, Any]:
        parser = ConfigParser(interpolation=None)

        log.info('Loading default configuration', config_file=str(self._default_config_file))
        parser.read(self._default_config_file)

        if config_file := arguments.get(self._config_file_argument):
            custom_config_file = Path(config_file)

            if os.path.exists(custom_config_file):
                log.info('Loading custom configuration', config_file=str(custom_config_file))
                parser.read(custom_config_file)
            else:
                try:
                    log.info('Creating custom configuration using default', config_file=str(custom_config_file))
                    copy_file(self._default_config_file, custom_config_file)
                except Exception as exception:
                    log.warn('Failed to create custom configuration file', error=str(exception))

        configuration = {}

        for section in parser.sections():
            configuration.update(dict(parser[section]))

        log.info('Loading command line arguments', arguments=arguments)
        configuration.update(arguments)

        log.info('Configuration loaded', configuration=configuration)

        return configuration
