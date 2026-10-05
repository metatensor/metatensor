"""
Check that all uses of ``pytest.raises`` and ``pytest.warns`` follow the pattern

.. code-block:: python

    message = "the expected error message"
    with pytest.raises(SomeError, match=message):
        ...

    message = "the expected warning message"
    with pytest.warns(SomeWarning, match=message):
        ...

i.e. that ``pytest.raises``/``pytest.warns`` are called with a single positional
argument (the exception/warning class or a tuple of classes), and a ``match`` keyword
argument which is a variable defined before the call (optionally wrapped in
``re.escape(...)``), and not an inline string.

Usage: python lint-pytest-match.py <files or directories>...
"""

import ast
import os
import sys


CHECKED_FUNCTIONS = {
    "raises": "exception",
    "warns": "warning",
}


def _pytest_function(node, aliases):
    """
    Get the name of the checked pytest function (``"raises"`` or ``"warns"``) called
    by ``node``, or ``None`` if this is not a call to one of these functions.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        if (
            func.attr in CHECKED_FUNCTIONS
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        ):
            return func.attr
    elif isinstance(func, ast.Name):
        return aliases.get(func.id)
    return None


def _match_variable(value):
    """
    Get the variable name used for ``match=...``, accepting both ``match=message``
    and ``match=re.escape(message)``. Returns ``None`` if the value is something else.
    """
    if isinstance(value, ast.Name):
        return value.id

    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "escape"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "re"
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Name)
    ):
        return value.args[0].id

    return None


def _defined_names(scope):
    """Get all names that are assigned to or are parameters in the given scope"""
    names = set()

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            names.add(arg.arg)
        if args.vararg is not None:
            names.add(args.vararg.arg)
        if args.kwarg is not None:
            names.add(args.kwarg.arg)

    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)

    return names


class PytestMatchChecker(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.errors = []
        # local names for functions imported with `from pytest import ...`
        self.aliases = {}
        # stack of scopes (module, functions, ...) containing the current node
        self.scopes = []

    def error(self, node, message):
        self.errors.append(
            f"{self.path}:{node.lineno}:{node.col_offset + 1}: {message}"
        )

    def visit_ImportFrom(self, node):
        if node.module == "pytest":
            for alias in node.names:
                if alias.name in CHECKED_FUNCTIONS:
                    self.aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def _visit_scope(self, node):
        self.scopes.append(node)
        self.generic_visit(node)
        self.scopes.pop()

    visit_Module = _visit_scope
    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope

    def visit_Call(self, node):
        function = _pytest_function(node, self.aliases)
        if function is not None:
            self.check_call(node, function)
        self.generic_visit(node)

    def check_call(self, node, function):
        kind = CHECKED_FUNCTIONS[function]
        function = f"pytest.{function}"

        if len(node.args) != 1 or any(isinstance(a, ast.Starred) for a in node.args):
            self.error(
                node,
                f"`{function}` should be called with exactly one positional "
                f"argument (the {kind} type), and used as a context manager",
            )

        match = None
        for keyword in node.keywords:
            if keyword.arg == "match":
                match = keyword
            else:
                self.error(
                    keyword.value,
                    f"`{function}` should only be called with the `match` "
                    "keyword argument",
                )

        if match is None:
            self.error(node, f"`{function}` is missing the `match=...` argument")
            return

        name = _match_variable(match.value)
        if name is None:
            self.error(
                match.value,
                "`match` should be a variable defined on a separate line, "
                f"e.g. `{function}({kind.title()}, match=message)`",
            )
            return

        if not any(name in _defined_names(scope) for scope in self.scopes):
            self.error(
                match.value,
                f"the `match` variable `{name}` is not defined in this scope",
            )


def check_file(path):
    with open(path, encoding="utf8") as fd:
        source = fd.read()

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: failed to parse file: {e.msg}"]

    checker = PytestMatchChecker(path)
    checker.visit(tree)
    return checker.errors


def python_files(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for file in sorted(files):
                    if file.endswith(".py"):
                        yield os.path.join(root, file)
        elif path.endswith(".py"):
            yield path


if __name__ == "__main__":
    errors = []
    for path in python_files(sys.argv[1:]):
        errors.extend(check_file(path))

    for error in errors:
        print(error)

    if errors:
        print(
            f"\nfound {len(errors)} invalid use(s) of `pytest.raises`/`pytest.warns`"
        )
        sys.exit(1)
