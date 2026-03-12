from enum import Enum

import netifaces
from context_logger import get_logger

log = get_logger('InterfaceResolver')


class AddressFamily(Enum):
    IPv4 = netifaces.AF_INET
    IPv6 = netifaces.AF_INET6
    MAC = netifaces.AF_LINK


class IInterfaceResolver:

    def resolve(self, interface: str, family: AddressFamily = AddressFamily.IPv4) -> str | None:
        raise NotImplementedError()


class InterfaceResolver(IInterfaceResolver):

    def resolve(self, interface: str, family: AddressFamily = AddressFamily.IPv4) -> str | None:
        interfaces = netifaces.interfaces()
        if interface in interfaces:
            inet_address = netifaces.ifaddresses(interface).get(family.value)
            if inet_address and len(inet_address) > 0:
                if address := inet_address[0].get('addr'):
                    return address
                else:
                    log.error('Address not found for interface', interface=interface, family=family.name)
            else:
                log.error('Address family not found for interface', interface=interface, family=family.name)
        else:
            log.error('Selected interface not found', interface=interface, interfaces=interfaces)

        return None
