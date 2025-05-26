from setuptools import setup, find_packages

setup(
    name='python-common-utility',
    description='Common utility packages for Python projects',
    author='Ferenc Nandor Janky & Attila Gombos',
    author_email='info@effective-range.com',
    packages=find_packages(exclude=['tests']),
    package_data={'common_utility': ['py.typed'], 'test_utility': ['py.typed']},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=[
        'requests',
        'pydantic',
        'jinja2',
        'python-context-logger@git+https://github.com/EffectiveRange/python-context-logger.git@latest',
    ],
)
