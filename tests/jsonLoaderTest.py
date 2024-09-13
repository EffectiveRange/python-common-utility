import unittest
from typing import Optional
from unittest import TestCase

from context_logger import setup_logging
from pydantic import ValidationError, BaseModel

from common_utility.jsonLoader import JsonLoader
from tests import TEST_RESOURCE_ROOT


class DeserializableClass1(BaseModel):
    attribute1: int
    attribute2: str
    attribute3: Optional[int] = None
    attribute4: list[str] = None


class DeserializableClass2(BaseModel):
    attribute1: Optional[DeserializableClass1]
    attribute2: list[DeserializableClass1]


class JsonLoaderTest(TestCase):
    TEST_instance_DIR = f'{TEST_RESOURCE_ROOT}/config'

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()

    def test_returns_instance_list_when_json_file_is_schema_valid(self):
        # Given
        json_loader = JsonLoader()

        # When
        instance_list = json_loader.load_list(f'{self.TEST_instance_DIR}/valid-list.json', DeserializableClass2)

        # Then
        self.assertEqual(2, len(instance_list))
        instance_1 = instance_list[0]
        self.assertEqual(instance_1.attribute1.attribute1, 1)
        self.assertEqual(instance_1.attribute1.attribute2, '1')
        self.assertEqual(instance_1.attribute1.attribute3, 1)
        self.assertEqual(instance_1.attribute1.attribute4, ['1', '1'])
        self.assertEqual(instance_1.attribute2[0].attribute1, 2)
        self.assertEqual(instance_1.attribute2[0].attribute2, '2')
        self.assertIsNone(instance_1.attribute2[0].attribute3)
        self.assertEqual(instance_1.attribute2[0].attribute4, ['2', '2'])
        self.assertEqual(instance_1.attribute2[1].attribute1, 3)
        self.assertEqual(instance_1.attribute2[1].attribute2, '3')
        self.assertEqual(instance_1.attribute2[1].attribute3, 3)
        self.assertEqual(instance_1.attribute2[1].attribute4, [])

        instance_2 = instance_list[1]
        self.assertEqual(instance_2.attribute1.attribute1, 4)
        self.assertEqual(instance_2.attribute1.attribute2, '4')
        self.assertIsNone(instance_2.attribute1.attribute3)
        self.assertEqual(instance_2.attribute1.attribute4, ['4', '4'])

    def test_returns_instance_when_json_file_is_schema_valid(self):
        # Given
        json_loader = JsonLoader()

        # When
        instance = json_loader.load(f'{self.TEST_instance_DIR}/valid-single.json', DeserializableClass2)

        # Then
        self.assertEqual(instance.attribute1.attribute1, 1)
        self.assertEqual(instance.attribute1.attribute2, '1')
        self.assertEqual(instance.attribute1.attribute3, 1)
        self.assertEqual(instance.attribute1.attribute4, ['1', '1'])
        self.assertEqual(instance.attribute2[0].attribute1, 2)
        self.assertEqual(instance.attribute2[0].attribute2, '2')
        self.assertIsNone(instance.attribute2[0].attribute3)
        self.assertEqual(instance.attribute2[0].attribute4, ['2', '2'])
        self.assertEqual(instance.attribute2[1].attribute1, 3)
        self.assertEqual(instance.attribute2[1].attribute2, '3')
        self.assertEqual(instance.attribute2[1].attribute3, 3)
        self.assertEqual(instance.attribute2[1].attribute4, [])

    def test_returns_instance_when_json_string_is_schema_valid(self):
        # Given
        json_loader = JsonLoader()

        # When
        instance = json_loader.load(
            '''
        {
          "attribute1": {
            "attribute1": 1,
            "attribute2": "1",
            "attribute3": 1,
            "attribute4": [
              "1",
              "1"
            ]
          },
          "attribute2": [
            {
              "attribute1": 2,
              "attribute2": "2",
              "attribute4": [
                "2",
                "2"
              ]
            }
          ]
        }
        ''',
            DeserializableClass2,
        )

        # Then
        self.assertEqual(instance.attribute1.attribute1, 1)
        self.assertEqual(instance.attribute1.attribute2, '1')
        self.assertEqual(instance.attribute1.attribute3, 1)
        self.assertEqual(instance.attribute1.attribute4, ['1', '1'])
        self.assertEqual(instance.attribute2[0].attribute1, 2)
        self.assertEqual(instance.attribute2[0].attribute2, '2')
        self.assertIsNone(instance.attribute2[0].attribute3)
        self.assertEqual(instance.attribute2[0].attribute4, ['2', '2'])

    def test_raises_error_when_json_file_is_schema_invalid(self):
        # Given
        json_loader = JsonLoader()

        # When
        self.assertRaises(
            ValidationError,
            json_loader.load_list,
            f'{self.TEST_instance_DIR}/invalid-list.json',
            DeserializableClass2,
        )

        # Then
        # Exception is raised

    def test_raises_error_when_root_type_is_invalid(self):
        # Given
        json_loader = JsonLoader()

        # When
        self.assertRaises(
            ValueError,
            json_loader.load,
            f'{self.TEST_instance_DIR}/package-config.list.json',
            DeserializableClass2,
        )

        # Then
        # Exception is raised


if __name__ == '__main__':
    unittest.main()
