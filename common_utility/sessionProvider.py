# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

from requests import Session


class ISessionProvider(object):

    def get_session(self) -> Session:
        raise NotImplementedError()


class SessionProvider(ISessionProvider):

    def get_session(self) -> Session:
        return Session()
