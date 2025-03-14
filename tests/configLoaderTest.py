import unittest
from unittest import TestCase

from context_logger import setup_logging

from common_utility import delete_file, ConfigLoader
from tests import TEST_RESOURCE_ROOT, TEST_FILE_SYSTEM_ROOT


class ConfigLoaderTest(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()
        delete_file(f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf')

    def test_load_config(self):
        # Given
        config_loader = ConfigLoader(TEST_RESOURCE_ROOT, 'config/example.conf')
        arguments = {
            'config_file': f'{TEST_RESOURCE_ROOT}/config/example.conf',
            'config_key1': 'new_value1',
        }

        # When
        result = config_loader.load(arguments)

        # Then
        self.assertEqual('new_value1', result['config_key1'])
        self.assertEqual('value2', result['config_key2'])
        self.assertEqual('example1', result['example_key1'])
        self.assertEqual('example2', result['example_key2'])

    def test_load_default_config_when_config_file_not_found(self):
        # Given
        config_loader = ConfigLoader(TEST_RESOURCE_ROOT, 'config/example.conf')
        arguments = {
            'config_file': f'{TEST_FILE_SYSTEM_ROOT}/etc/example.conf',
            'example_key1': 'new_example1',
        }

        # When
        result = config_loader.load(arguments)

        # Then
        self.assertEqual('value1', result['config_key1'])
        self.assertEqual('value2', result['config_key2'])
        self.assertEqual('new_example1', result['example_key1'])
        self.assertEqual('example2', result['example_key2'])


if __name__ == '__main__':
    unittest.main()
