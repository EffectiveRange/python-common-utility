import unittest
from time import sleep
from unittest import TestCase
from unittest.mock import MagicMock

from context_logger import setup_logging

from common_utility import ReusableTimer
from test_utility import wait_for_assertion


class ReusableTimerTest(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()

    def test_start(self):
        # Given
        mock = MagicMock()
        timer = ReusableTimer()

        # When
        timer.start(1, mock.test_method, args=[1], kwargs={'b': 2, 'c': 3})

        # Then
        self.assertTrue(timer.is_alive())
        wait_for_assertion(1.1, mock.test_method.assert_called_once_with, 1, b=2, c=3)

    def test_restart(self):
        # Given
        mock = MagicMock()
        timer = ReusableTimer()
        timer.start(1, mock.test_method, args=[1], kwargs={'b': 2, 'c': 3})
        wait_for_assertion(1.1, mock.test_method.assert_called_once_with, 1, b=2, c=3)
        mock.reset_mock()

        # When
        timer.restart()

        # Then
        self.assertTrue(timer.is_alive())
        wait_for_assertion(1.1, mock.test_method.assert_called_once_with, 1, b=2, c=3)

    def test_cancel(self):
        # Given
        mock = MagicMock()
        timer = ReusableTimer()
        timer.start(1, mock.test_method, args=[1], kwargs={'b': 2, 'c': 3})

        # When
        timer.cancel()

        # Then
        self.assertFalse(timer.is_alive())
        sleep(1.1)
        mock.test_method.assert_not_called()

    def test_start_again(self):
        # Given
        mock = MagicMock()
        timer = ReusableTimer()
        timer.start(1, mock.test_method, args=[1], kwargs={'b': 2, 'c': 3})
        wait_for_assertion(1.1, mock.test_method.assert_called_once_with, 1, b=2, c=3)
        mock.reset_mock()

        # When
        timer.start(1, mock.test_method, args=[2], kwargs={'b': 4, 'c': 6})

        # Then
        self.assertTrue(timer.is_alive())
        wait_for_assertion(1.1, mock.test_method.assert_called_once_with, 2, b=4, c=6)


if __name__ == '__main__':
    unittest.main()
