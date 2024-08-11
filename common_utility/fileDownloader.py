# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import os
from typing import Optional
from urllib.parse import urlparse

from context_logger import get_logger
from requests import Response

from common_utility import ISessionProvider

log = get_logger('FileDownloader')


class IFileDownloader(object):

    def download(
        self,
        file_url: str,
        file_name: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        skip_if_exists: bool = True,
        chunk_size: int = 1000 * 1000,
    ) -> str:
        raise NotImplementedError()


class FileDownloader(IFileDownloader):

    def __init__(self, session_provider: ISessionProvider, download_location: str) -> None:
        self._session_provider = session_provider
        self._download_location = download_location

    def download(
        self,
        file_url: str,
        file_name: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        skip_if_exists: bool = True,
        chunk_size: int = 1000 * 1000,
    ) -> str:
        if not urlparse(file_url).scheme:
            return self._check_local_file(file_url)

        file_path = self._get_download_path(file_url, file_name)

        if skip_if_exists and os.path.isfile(file_path):
            log.info('File already exists, skipping download', file=file_path)
            return file_path

        headers = headers if headers else dict()

        log.info('Downloading file', url=file_url, file_name=file_name, headers=list(headers.keys()))

        response = self._send_request(file_url, headers)

        self._download_file(response, file_path, chunk_size)

        log.info('Downloaded file', file=file_path)

        return file_path

    def _check_local_file(self, file_url: str) -> str:
        file_path = os.path.abspath(file_url)
        if os.path.isfile(file_path):
            log.info('Local file path provided, skipping download', file=file_path)
            return file_path
        else:
            log.error('Local file does not exist', file=file_path)
            raise ValueError('Local file does not exist')

    def _send_request(self, file_url: str, headers: dict[str, str]) -> Response:
        with self._session_provider.get_session() as session:
            response = session.get(file_url, stream=True, headers=headers)

        if response.status_code != 200:
            log.error('Failed to download file', url=file_url, status_code=response.status_code, reason=response.reason)
            raise ValueError('Failed to download file')

        return response

    def _get_download_path(self, file_url: str, file_name: Optional[str]) -> str:
        if not file_name and '/' in file_url:
            file_name = file_url.split('/')[-1]

        return f'{self._download_location}/{file_name}'

    def _download_file(self, response: Response, file_path: str, chunk_size: int) -> None:
        if not os.path.exists(self._download_location):
            os.makedirs(self._download_location)

        with open(file_path, 'wb') as asset_file:
            for chunk in response.iter_content(chunk_size):
                asset_file.write(chunk)
