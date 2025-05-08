import unittest
from unittest import TestCase
from unittest.mock import patch

from context_logger import setup_logging

from common_utility import TokenBucketLimiter


class RateLimiterTest(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()

    @patch('time.monotonic', side_effect=[0.0, 100.0, 100.9, 101.2])
    def test_acquire_when_not_exceeds_limit(self, mock_monotonic):
        # Given
        rate_limiter = TokenBucketLimiter(10, 1, 2)

        rate_limiter.acquire(10)
        rate_limiter.acquire(10)

        # When
        result = rate_limiter.acquire(10)

        # Then
        self.assertTrue(result)

    @patch('time.monotonic', side_effect=[0.0, 100.0, 100.9, 101.2])
    def test_acquire_when_exceeds_limit(self, mock_monotonic):
        # Given
        rate_limiter = TokenBucketLimiter(10, 1, 2)

        rate_limiter.acquire(12)
        rate_limiter.acquire(12)

        # When
        result = rate_limiter.acquire(12)

        # Then
        self.assertFalse(result)

    @patch('time.monotonic', side_effect=[0.0, 100.0, 130.0, 160.0])
    def test_acquire_when_not_exceeds_limit_and_minute_based(self, mock_monotonic):
        # Given
        rate_limiter = TokenBucketLimiter(10, 60, 2)

        rate_limiter.acquire(10)
        rate_limiter.acquire(10)

        # When
        result = rate_limiter.acquire(10)

        # Then
        self.assertTrue(result)

    @patch('time.monotonic', side_effect=[0.0, 100.0, 130.0, 160.0])
    def test_acquire_when_exceeds_limit_and_minute_based(self, mock_monotonic):
        # Given
        rate_limiter = TokenBucketLimiter(10, 60, 2)

        rate_limiter.acquire(12)
        rate_limiter.acquire(12)

        # When
        result = rate_limiter.acquire(12)

        # Then
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
