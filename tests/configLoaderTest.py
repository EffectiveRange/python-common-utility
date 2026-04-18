import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from context_logger import setup_logging

from common_utility import delete_file, ConfigLoader, copy_file
from tests import TEST_RESOURCE_ROOT, TEST_FILE_SYSTEM_ROOT

DEFAULT_CONFIG_FILE = f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf.default'


class ConfigLoaderTest(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)
        copy_file(f'{TEST_RESOURCE_ROOT}/config/example.conf.default', DEFAULT_CONFIG_FILE)

    def setUp(self):
        print()
        delete_file(f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf')
        delete_file(f'{TEST_FILE_SYSTEM_ROOT}/etc/example.types.conf')
        delete_file(f'{TEST_FILE_SYSTEM_ROOT}/etc/example.types.invalid.conf')

    def _create_argument_parser(self) -> ArgumentParser:
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--config', default=None)
        argument_parser.add_argument('--config-key1', default=None)
        argument_parser.add_argument('--config-key2', default=None)
        argument_parser.add_argument('--example-key1', default=None)
        argument_parser.add_argument('--example-key2', default=None)
        return argument_parser

    def test_load_config_when_default_config_file_could_not_be_loaded(self):
        # Given
        config_loader = ConfigLoader(Path('invalid/path/example.conf.default'))
        argument_parser = self._create_argument_parser()

        # When
        with patch.object(sys, 'argv', ['test', '--config-key1', 'new_value1']):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('new_value1', result.config_key1)

    def test_load_config_when_no_custom_config_file_specified(self):
        # Given
        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = self._create_argument_parser()

        # When
        with patch.object(sys, 'argv', ['test', '--config-key1', 'new_value1']):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('new_value1', result.config_key1)
        self.assertEqual('value2', result.config_key2)
        self.assertEqual('example1', result.example_key1)
        self.assertEqual('example2', result.example_key2)

    def test_load_config_when_custom_config_file_specified(self):
        # Given
        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = self._create_argument_parser()
        config_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf'

        copy_file(f'{TEST_RESOURCE_ROOT}/config/example.conf', config_file)

        # When
        with patch.object(sys, 'argv', ['test', '--config', config_file, '--example-key1', 'new_example1']):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('value1', result.config_key1)
        self.assertEqual('value3', result.config_key2)
        self.assertEqual('new_example1', result.example_key1)
        self.assertEqual('example4', result.example_key2)

    def test_load_config_when_custom_config_file_could_not_be_loaded(self):
        # Given
        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = self._create_argument_parser()
        # When
        with patch.object(sys, 'argv',
                          ['test', '--config', 'invalid/path/example.conf', '--example-key1', 'new_example1']):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('value2', result.config_key2)
        self.assertEqual('new_example1', result.example_key1)
        self.assertEqual('example2', result.example_key2)

    def test_load_config_when_parser_default_values_defined_but_not_passed(self):
        # Given
        config_loader = ConfigLoader(Path(TEST_RESOURCE_ROOT) / 'config' / 'example.conf.default')
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--config', default=None)
        argument_parser.add_argument('--config-key1', default='cli_default_value1')
        argument_parser.add_argument('--example-key1', default='cli_default_example1')
        config_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf'

        copy_file(f'{TEST_RESOURCE_ROOT}/config/example.conf', config_file)

        # When
        with patch.object(sys, 'argv', ['test', '--config', config_file]):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('value1', result.config_key1)
        self.assertEqual('example3', result.example_key1)

    def test_load_config_when_short_option_cli_override_is_passed(self):
        # Given
        config_loader = ConfigLoader(Path(TEST_RESOURCE_ROOT) / 'config' / 'example.conf.default')
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--config', default=None)
        argument_parser.add_argument('--example-key2', '-e2', default=None)
        config_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf'

        copy_file(f'{TEST_RESOURCE_ROOT}/config/example.conf', config_file)

        # When
        with patch.object(sys, 'argv', ['test', '--config', config_file, '-e2', 'new_example2']):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('new_example2', result.example_key2)

    def test_load_config_when_long_option_cli_override_passed_using_equal_sign(self):
        # Given
        config_loader = ConfigLoader(Path(TEST_RESOURCE_ROOT) / 'config' / 'example.conf.default')
        argument_parser = self._create_argument_parser()

        # When
        with patch.object(sys, 'argv', ['test', '--config-key1=new_value1']):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('value1', result.config_key1)

    def test_get_cli_overrides_when_no_cli_arguments(self):
        # Given
        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = self._create_argument_parser()
        arguments = argument_parser.parse_args([])

        # When
        with patch.object(sys, 'argv', ['test']):
            result = config_loader._get_cli_overrides(argument_parser, arguments)

        # Then
        self.assertEqual({}, result)

    def test_get_cli_overrides_when_argument_has_no_option_string(self):
        # Given
        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--config', default=None)
        argument_parser.add_argument('input_file')
        arguments = argument_parser.parse_args(['input.txt'])

        # When
        with patch.object(sys, 'argv', ['test', 'input.txt']):
            result = config_loader._get_cli_overrides(argument_parser, arguments)

        # Then
        self.assertEqual({}, result)

    def test_load_config_when_type_values_present_then_sanitize(self):
        # Given
        config_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/example.types.conf'
        Path(config_file).write_text('[types]\nfeature_enabled = true\nretry_count = 7\ntimeout = 1.5\n',
                                     encoding='utf-8')

        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--config', default=None)
        argument_parser.add_argument('--feature-enabled', default=False)
        argument_parser.add_argument('--retry-count', default=0)
        argument_parser.add_argument('--timeout', default=0.0)

        # When
        with patch.object(sys, 'argv', ['test', '--config', config_file]):
            result = config_loader.load(argument_parser)

        # Then
        self.assertTrue(result.feature_enabled)
        self.assertEqual(7, result.retry_count)
        self.assertEqual(1.5, result.timeout)

    def test_load_config_when_invalid_numeric_values_present_then_keep_original(self):
        # Given
        config_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/example.types.invalid.conf'
        Path(config_file).write_text('[types]\nretry_count = invalid\ntimeout = invalid\n', encoding='utf-8')

        config_loader = ConfigLoader(Path(DEFAULT_CONFIG_FILE))
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--config', default=None)
        argument_parser.add_argument('--retry-count', default=0)
        argument_parser.add_argument('--timeout', default=0.0)

        # When
        with patch.object(sys, 'argv', ['test', '--config', config_file]):
            result = config_loader.load(argument_parser)

        # Then
        self.assertEqual('invalid', result.retry_count)
        self.assertEqual('invalid', result.timeout)


if __name__ == '__main__':
    unittest.main()
