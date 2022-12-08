# TODO: use importlib.metadata instead of pkg_resources once we drop Python 3.7
from pkg_resources import get_distribution


__version__ = get_distribution("equistore").version
