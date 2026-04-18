# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import sys
from argparse import ArgumentParser, Action, Namespace
from configparser import ConfigParser
from pathlib import Path
from typing import Any, cast

from context_logger import get_logger


class IConfigLoader(object):

    def load(self, argument_parser: ArgumentParser) -> Namespace:
        raise NotImplementedError()


class ConfigLoader(IConfigLoader):

    def __init__(self, default_config_file: Path) -> None:
        self._config_parser = ConfigParser(interpolation=None)
        self._default_config_file = default_config_file
        self.log = get_logger(type(self).__name__)

    def load(self, argument_parser: ArgumentParser) -> Namespace:
        arguments = argument_parser.parse_known_args()[0]

        self.log.info('Loading default configuration', config_file=str(self._default_config_file))
        loaded = self._config_parser.read(self._default_config_file)

        if str(self._default_config_file) not in loaded:
            self.log.warn('Default configuration could not be loaded', config_file=str(self._default_config_file))

        if custom_config_file := arguments.config:
            custom_config_file = Path(custom_config_file)

            self.log.info('Loading custom configuration', config_file=str(custom_config_file))
            loaded = self._config_parser.read(custom_config_file)

            if str(custom_config_file) not in loaded:
                self.log.warn('Custom configuration could not be loaded', config_file=str(custom_config_file))

        configuration = dict(vars(arguments))

        for section in self._config_parser.sections():
            configuration.update(dict(self._config_parser[section]))

        self.log.info('Loading command line arguments', arguments=vars(arguments))
        cli_overrides = self._get_cli_overrides(argument_parser, arguments)
        configuration.update(cli_overrides)

        self._sanitize_config(argument_parser, configuration)

        self.log.info('Configuration loaded', configuration=configuration)

        return Namespace(**configuration)

    def _get_cli_overrides(self, parser: ArgumentParser, arguments: Namespace) -> dict[str, Any]:
        cli_overrides: dict[str, Any] = {}
        argv_tokens = set(sys.argv[1:])

        if not argv_tokens:
            return cli_overrides

        for action in parser._actions:
            if action.dest == 'help':
                continue

            if action.option_strings:  # has --flag or -f
                if any(opt in argv_tokens for opt in action.option_strings):
                    cli_overrides[action.dest] = getattr(arguments, action.dest)

        return cli_overrides

    def _sanitize_config(self, parser: ArgumentParser, config: dict[str, Any]) -> None:
        for config_key in config:
            action = self._find_action(parser, config_key)
            if action is None or action.default is None:
                continue

            if isinstance(action.default, bool):
                self._convert_bool(config, config_key)
            elif isinstance(action.default, int):
                self._convert_int(config, config_key)
            elif isinstance(action.default, float):
                self._convert_float(config, config_key)

        for config_key in config:
            self.log.debug('Config', key=config_key, value=config[config_key], type=type(config[config_key]))

    def _find_action(self, parser: ArgumentParser, config_key: str) -> Action:
        return next((a for a in parser._actions if a.dest == config_key), cast(Action, cast(object, None)))

    def _convert_bool(self, config: dict[str, Any], config_key: str) -> None:
        config[config_key] = str(config[config_key]).lower() in ('true', '1', 'yes')

    def _convert_int(self, config: dict[str, Any], config_key: str) -> None:
        try:
            config[config_key] = int(config[config_key])
        except (TypeError, ValueError):
            pass

    def _convert_float(self, config: dict[str, Any], config_key: str) -> None:
        try:
            config[config_key] = float(config[config_key])
        except (TypeError, ValueError):
            pass
