import unittest
from unittest import TestCase
from unittest.mock import patch

from context_logger import setup_logging

from common_utility import InterfaceResolver, AddressFamily


class InterfaceResolverTest(TestCase):
    resolver = InterfaceResolver()

    @classmethod
    def setUpClass(cls):
        setup_logging('python-common-utility', 'DEBUG', warn_on_overwrite=False)

    def setUp(self):
        print()

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_ipv4_address(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)

        # When
        address = self.resolver.resolve('lo')

        # Then
        self.assertEqual('127.0.0.1', address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_ipv6_address(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)

        # When
        address = self.resolver.resolve('lo', AddressFamily.IPv6)

        # Then
        self.assertEqual('::1', address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_mac_address(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)

        # When
        address = self.resolver.resolve('lo', AddressFamily.MAC)

        # Then
        self.assertEqual('00:00:00:00:00:00', address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_none_when_interface_not_exists(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)
        mock_netifaces.interfaces.return_value = ['lo', 'wlan0']

        # When
        address = self.resolver.resolve('eth0')

        # Then
        self.assertIsNone(address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_none_when_address_family_not_exists_for_interface(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)
        mock_netifaces.ifaddresses.return_value = {
            10: [{'addr': '::1'}],
            17: [{'addr': '00:00:00:00:00:00'}],
        }

        # When
        address = self.resolver.resolve('lo')

        # Then
        self.assertIsNone(address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_none_when_address_not_exists_for_address_family(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)
        mock_netifaces.ifaddresses.return_value = {
            2: [],
        }

        # When
        address = self.resolver.resolve('wlan0')

        # Then
        self.assertIsNone(address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_none_when_addr_key_not_exists_for_address(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)
        mock_netifaces.ifaddresses.return_value = {
            2: [{}],
        }

        # When
        address = self.resolver.resolve('eth0')

        # Then
        self.assertIsNone(address)

    @patch('common_utility.interfaceResolver.netifaces')
    def test_returns_none_when_addr_value_not_exists_for_address(self, mock_netifaces):
        # Given
        self._setup_mocks(mock_netifaces)
        mock_netifaces.ifaddresses.return_value = {
            2: [{'addr': ''}],
        }

        # When
        address = self.resolver.resolve('eth0')

        # Then
        self.assertIsNone(address)

    def _setup_mocks(self, mock_netifaces):
        mock_netifaces.interfaces.return_value = ['lo', 'eth0', 'wlan0']
        mock_netifaces.AF_INET = 2
        mock_netifaces.AF_INET6 = 10
        mock_netifaces.AF_INET6 = 17
        mock_netifaces.ifaddresses.return_value = {
            2: [{'addr': '127.0.0.1'}],
            10: [{'addr': '::1'}],
            17: [{'addr': '00:00:00:00:00:00'}],
        }


if __name__ == '__main__':
    unittest.main()
