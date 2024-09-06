# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/EffectiveRange/python-common-utility/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| common\_utility/\_\_init\_\_.py    |        5 |        0 |        0 |        0 |    100% |           |
| common\_utility/fileDownloader.py  |       50 |        0 |       18 |        3 |     96% |75->78, 94->exit, 95->94 |
| common\_utility/jsonLoader.py      |       30 |        1 |       10 |        2 |     92% |40->exit, 50 |
| common\_utility/reusableTimer.py   |       37 |        0 |       14 |        6 |     88% |47->54, 57->exit, 58->57, 71->exit, 72->71, 84->exit |
| common\_utility/sessionProvider.py |        6 |        1 |        0 |        0 |     83% |        17 |
|                          **TOTAL** |  **128** |    **2** |   **42** |   **11** | **92%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/EffectiveRange/python-common-utility/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/EffectiveRange/python-common-utility/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/EffectiveRange/python-common-utility/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/EffectiveRange/python-common-utility/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FEffectiveRange%2Fpython-common-utility%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/EffectiveRange/python-common-utility/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.