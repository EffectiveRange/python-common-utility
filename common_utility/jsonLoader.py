# SPDX-FileCopyrightText: 2024 Ferenc Nandor Janky <ferenj@effective-range.com>
# SPDX-FileCopyrightText: 2024 Attila Gombos <attila.gombos@effective-range.com>
# SPDX-License-Identifier: MIT

import json
import os
from typing import TypeVar, Type, Union, List, Callable, Any

from context_logger import get_logger
from pydantic import BaseModel

log = get_logger('JsonLoader')

T = TypeVar('T', bound=BaseModel)


class IJsonLoader(object):

    def load(self, json_file_path: str, model: Type[T]) -> T:
        raise NotImplementedError()

    def load_list(self, json_file_path: str, model: Type[T]) -> List[T]:
        raise NotImplementedError()


class JsonLoader(IJsonLoader):

    def load(self, json_data: str, model: Type[T]) -> T:
        data = self._load_data(json_data)

        return self._validate(data, dict, lambda: model(**data))  # type: ignore

    def load_list(self, json_data: str, model: Type[T]) -> List[T]:
        data = self._load_data(json_data)

        return self._validate(data, list, lambda: [model(**item) for item in data])  # type: ignore

    def _load_data(self, json_data: str) -> Any:
        if os.path.isfile(json_data):
            with open(json_data, 'r') as json_file:
                return json.load(json_file)
        else:
            return json.loads(json_data)

    def _validate(self, data: Any, root_type: Any, deserialize: Callable[[], Union[T, List[T]]]) -> Union[T, List[T]]:
        try:
            if isinstance(data, root_type):
                return deserialize()
            else:
                raise ValueError(f'Unexpected JSON root type: {data.__class__.__name__}')
        except ValueError as error:
            log.error('Failed to load JSON file', error=error)
            raise error
