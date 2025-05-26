# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from context_logger import get_logger
from requests import Response

from common_utility import ISessionProvider, create_directory, copy_file

log = get_logger('FileDownloader')


class IFileDownloader(object):

    def download(self, file_url: str, file_name: Optional[str] = None, sub_dir: Optional[Union[str, Path]] = None,
                 headers: Optional[dict[str, str]] = None, skip_if_exists: bool = True, chunk_size: int = 1000 * 1000
                 ) -> Path:
        raise NotImplementedError()

    def download_and_copy(self, file_url: str, sub_dirs: list[Union[str, Path]], file_name: Optional[str] = None,
                          headers: Optional[dict[str, str]] = None, skip_if_exists: bool = True,
                          chunk_size: int = 1000 * 1000) -> list[Path]:
        raise NotImplementedError()


class FileDownloader(IFileDownloader):

    def __init__(self, session_provider: ISessionProvider, download_location: Path) -> None:
        self._session_provider = session_provider
        self._download_location = download_location

    def download(self, file_url: str, file_name: Optional[str] = None, sub_dir: Optional[Union[str, Path]] = None,
                 headers: Optional[dict[str, str]] = None, skip_if_exists: bool = True, chunk_size: int = 1000 * 1000
                 ) -> Path:
        if not urlparse(file_url).scheme:
            return self._check_local_file(file_url)

        file_path = self._get_target_path(file_url, file_name, sub_dir)

        if skip_if_exists and os.path.isfile(file_path):
            log.info('File already exists, skipping download', file=str(file_path))
            return file_path

        headers = headers if headers else dict()

        log.info('Downloading file', url=file_url, file_name=file_name, headers=list(headers.keys()))

        response = self._send_request(file_url, headers)

        self._download_file(response, file_path, chunk_size)

        log.info('Downloaded file', file=str(file_path))

        return file_path

    def download_and_copy(self, file_url: str, sub_dirs: list[Union[str, Path]], file_name: Optional[str] = None,
                          headers: Optional[dict[str, str]] = None, skip_if_exists: bool = True,
                          chunk_size: int = 1000 * 1000) -> list[Path]:
        if not sub_dirs:
            raise ValueError('At least one sub directory must be provided')

        downloaded_files = [self.download(file_url, file_name, sub_dirs[0], headers, skip_if_exists, chunk_size)]

        for sub_dir in sub_dirs[1:]:
            file_path = self._get_target_path(file_url, file_name, sub_dir)

            if skip_if_exists and os.path.isfile(file_path):
                log.info('File already exists, skipping copy', file=str(file_path))
            else:
                copy_file(downloaded_files[0], file_path)
                log.info('Copied downloaded file', file=str(file_path))

            downloaded_files.append(file_path)

        return downloaded_files

    def _check_local_file(self, file_url: str) -> Path:
        file_path = os.path.abspath(file_url)
        if os.path.isfile(file_path):
            log.info('Local file path provided, skipping download', file=str(file_path))
            return Path(file_path)
        else:
            log.error('Local file does not exist', file=str(file_path))
            raise ValueError('Local file does not exist')

    def _send_request(self, file_url: str, headers: dict[str, str]) -> Response:
        with self._session_provider.get_session() as session:
            response = session.get(file_url, stream=True, headers=headers)

        if response.status_code != 200:
            log.error('Failed to download file', url=file_url, status_code=response.status_code, reason=response.reason)
            raise ValueError('Failed to download file')

        return response

    def _get_target_path(self, file_url: str, file_name: Optional[str],
                         sub_dir: Optional[Union[str, Path]] = None) -> Path:
        if not file_name:
            file_name = file_url.split('/')[-1]

        download_dir = self._download_location

        if sub_dir:
            download_dir = download_dir / sub_dir
            create_directory(download_dir)

        return download_dir / file_name

    def _download_file(self, response: Response, file_path: Path, chunk_size: int) -> None:
        create_directory(self._download_location)

        with open(file_path, 'wb') as asset_file:
            for chunk in response.iter_content(chunk_size):
                asset_file.write(chunk)
