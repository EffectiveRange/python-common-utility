from setuptools import setup

setup(
    name='python-common-utility',
    version='1.0.0',
    description='Common utility packages for Python projects',
    author='Ferenc Nandor Janky & Attila Gombos',
    author_email='info@effective-range.com',
    packages=['common_utility', 'test_utility'],
    package_data={'common_utility': ['py.typed'], 'test_utility': ['py.typed']},
    install_requires=[
        'requests',
        'pydantic',
        'jinja2',
        'python-context-logger@git+https://github.com/EffectiveRange/python-context-logger.git@latest',
    ],
)
