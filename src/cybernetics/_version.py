from importlib.metadata import version

# The PyPI *distribution* name (import package is ``cybernetics``); importlib
# metadata is keyed on the distribution, so this must match pyproject ``name``.
__title__ = "cybernetic-physics"
__version__ = version(__title__)
