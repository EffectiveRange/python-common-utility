import os.path
import unittest
from pathlib import Path
from unittest import TestCase

from context_logger import setup_logging

from common_utility import delete_file, ConfigLoader, copy_file
from tests import TEST_RESOURCE_ROOT, TEST_FILE_SYSTEM_ROOT


class ConfigLoaderTest(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()
        delete_file(f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf')

    def test_load_config_when_custom_configuration_not_exists(self):
        # Given
        config_loader = ConfigLoader(Path(TEST_RESOURCE_ROOT) / 'config' / 'example.default.conf')
        arguments = {
            'config_file': f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf',
            'config_key1': 'new_value1',
        }

        # When
        result = config_loader.load(arguments)

        # Then
        self.assertTrue(os.path.exists(f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf'))
        self.assertEqual('new_value1', result['config_key1'])
        self.assertEqual('value2', result['config_key2'])
        self.assertEqual('example1', result['example_key1'])
        self.assertEqual('example2', result['example_key2'])

    def test_load_config_when_custom_configuration_exists(self):
        # Given
        config_loader = ConfigLoader(Path(TEST_RESOURCE_ROOT) / 'config' / 'example.default.conf')
        arguments = {
            'config_file': f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf',
            'example_key1': 'new_example1',
        }

        copy_file(f'{TEST_RESOURCE_ROOT}/config/example.conf', f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf')

        # When
        result = config_loader.load(arguments)

        # Then
        self.assertEqual('value1', result['config_key1'])
        self.assertEqual('value3', result['config_key2'])
        self.assertEqual('new_example1', result['example_key1'])
        self.assertEqual('example4', result['example_key2'])

    def test_load_config_when_fail_to_create_custom_configuration(self):
        # Given
        config_loader = ConfigLoader(Path(TEST_RESOURCE_ROOT) / 'config' / 'example.default.conf')
        arguments = {
            'config_file': '/invalid/path/to/example.conf',
            'config_key1': 'new_value1',
        }

        # When
        result = config_loader.load(arguments)

        # Then
        self.assertEqual('new_value1', result['config_key1'])
        self.assertEqual('value2', result['config_key2'])
        self.assertEqual('example1', result['example_key1'])
        self.assertEqual('example2', result['example_key2'])


if __name__ == '__main__':
    unittest.main()
