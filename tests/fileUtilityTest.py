import os
import unittest
from unittest import TestCase

from context_logger import setup_logging

from common_utility import create_directory, delete_directory, create_file, delete_file, copy_file, append_file, \
    replace_in_file, is_file_matches_pattern, is_file_contains_lines, render_template_file
from tests import TEST_FILE_SYSTEM_ROOT


class FileUtilityTest(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()

    def test_create_directory(self):
        # Given
        directory = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range'

        # When
        create_directory(directory)

        # Then
        self.assertTrue(os.path.exists(directory))

        # Clean up
        delete_directory(directory)

    def test_create_directory_when_exists(self):
        # Given
        directory = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range'
        create_directory(directory)

        # When
        create_directory(directory)

        # Then
        self.assertTrue(os.path.exists(directory))

        # Clean up
        delete_directory(directory)

    def test_delete_directory(self):
        # Given
        directory = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range'
        create_directory(directory)

        # When
        delete_directory(directory)

        # Then
        self.assertFalse(os.path.exists(directory))

    def test_delete_directory_when_not_exists(self):
        # Given
        directory = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range'

        # When
        delete_directory(directory)

        # Then
        self.assertFalse(os.path.exists(directory))

    def test_create_file(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'

        # When
        create_file(file_path, 'Hello, World!')

        # Then
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, 'r') as file:
            content = file.read()
            self.assertEqual(content, 'Hello, World!')

        # Clean up
        delete_file(file_path)

    def test_delete_file(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'

        # When
        create_file(file_path, 'Hello, World!')
        delete_file(file_path)

        # Then
        self.assertFalse(os.path.exists(file_path))

    def test_delete_file_when_link(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, 'Hello, World!')
        link_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test_link.txt'
        os.symlink(file_path, link_path)

        # When
        delete_file(file_path)

        # Then
        self.assertFalse(os.path.exists(file_path))

    def test_copy_file(self):
        # Given
        source_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/source.txt'
        destination_file = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/destination.txt'
        create_file(source_file, 'Hello, World!')

        # When
        copy_file(source_file, destination_file)

        # Then
        self.assertTrue(os.path.exists(destination_file))
        with open(destination_file, 'r') as file:
            content = file.read()
            self.assertEqual(content, 'Hello, World!')

        # Clean up
        delete_file(source_file)
        delete_file(destination_file)

    def test_append_file(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, '')

        # When
        append_file(file_path, 'Hello, World!')

        # Then
        with open(file_path, 'r') as file:
            content = file.read()
            self.assertEqual('Hello, World!\n', content)

        # Clean up
        delete_file(file_path)

    def test_is_file_matches_pattern(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, 'Hello, World!')

        # When
        result = is_file_matches_pattern(file_path, 'Hello, .*')

        # Then
        self.assertTrue(result)

        # Clean up
        delete_file(file_path)

    def test_is_file_matches_pattern_when_not_matches(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, 'Hi, World!')

        # When
        result = is_file_matches_pattern(file_path, 'Hello, .*')

        # Then
        self.assertFalse(result)

        # Clean up
        delete_file(file_path)

    def test_is_file_matches_pattern_when_not_exists(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'

        # When
        result = is_file_matches_pattern(file_path, 'Hello, .*')

        # Then
        self.assertFalse(result)

    def test_is_file_contains_lines(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, """Hello, World!
This is a test file.
It contains multiple lines.""")

        # When
        result = is_file_contains_lines(file_path,
                                        ['Hello, World!', 'This is a test file.', 'It contains multiple lines.'])

        # Then
        self.assertTrue(result)

        # Clean up
        delete_file(file_path)

    def test_is_file_contains_lines_when_not_contains(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, """Hello, World!
This is a test file.
It contains multiple lines.""")

        # When
        result = is_file_contains_lines(file_path,
                                        ['Hi, World!', 'This is a test file.', 'It contains multiple lines.'])

        # Then
        self.assertFalse(result)

        # Clean up
        delete_file(file_path)

    def test_is_file_contains_lines_when_not_exists(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'

        # When
        result = is_file_contains_lines(file_path,
                                        ['Hello, World!', 'This is a test file.', 'It contains multiple lines.'])

        # Then
        self.assertFalse(result)

    def test_render_template_file(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, 'Hello, {{ name }}!')

        # When
        result = render_template_file(file_path, {'name': 'World'})

        # Then
        self.assertEqual(result, 'Hello, World!\n')

        # Clean up
        delete_file(file_path)

    def test_replace_in_file(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, 'Hello, World!')

        # When
        replace_in_file(file_path, 'World', 'Python')

        # Then
        with open(file_path, 'r') as file:
            content = file.read()
            self.assertEqual('Hello, Python!', content)

        # Clean up
        delete_file(file_path)

    def test_replace_in_file_when_not_exists(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'

        # When
        replace_in_file(file_path, 'World', 'Python')

        # Then
        # No exception should be raised

    def test_replace_in_file_with_regex(self):
        # Given
        file_path = f'{TEST_FILE_SYSTEM_ROOT}/etc/effective-range/test.txt'
        create_file(file_path, 'Hello, World!')

        # When
        replace_in_file(file_path, 'W([A-Za-z]+)', 'Python')

        # Then
        with open(file_path, 'r') as file:
            content = file.read()
            self.assertEqual('Hello, Python!', content)

        # Clean up
        delete_file(file_path)


if __name__ == '__main__':
    unittest.main()
