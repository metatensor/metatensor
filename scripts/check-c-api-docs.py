"""
A small script checking that all the C API functions are documented
"""
import os
import sys

from pycparser import c_ast, parse_file


ROOT = os.path.join(os.path.dirname(__file__), "..")
C_API_DOCS = os.path.join(ROOT, "docs", "src", "core", "reference", "c")
FAKE_INCLUDES = os.path.join(ROOT, "python", "scripts", "include")
METATENSOR_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatensor-core", "include", "metatensor.h")
)


ERRORS = 0


def error(message):
    global ERRORS
    ERRORS += 1
    print(message)


def documented_functions():
    functions = []

    for root, _, paths in os.walk(C_API_DOCS):
        for path in paths:
            with open(os.path.join(root, path), encoding="utf8") as fd:
                for line in fd:
                    if line.startswith(".. doxygenfunction::"):
                        name = line.split()[2]
                        functions.append(name)

    return functions


def functions_in_outline():
    # function from the "miscellaneous" section of the docs don't require an outline
    # (since they are not related to a specific struct type)
    functions = [
        "mts_version",
        "mts_last_error",
        "mts_disable_panic_printing",
        "mts_get_data_origin",
        "mts_register_data_origin",
        "mts_tensormap_load",
        "mts_tensormap_load_buffer",
        "mts_tensormap_save",
        "mts_tensormap_save_buffer",
    ]

    for root, _, paths in os.walk(C_API_DOCS):
        for path in paths:
            with open(os.path.join(root, path), encoding="utf8") as fd:
                for line in fd:
                    if ":c:func:" in line:
                        name = line.split("`")[1]
                        functions.append(name)
    return functions


def all_functions():
    cpp_args = ["-E", "-I", FAKE_INCLUDES]
    ast = parse_file(METATENSOR_HEADER, use_cpp=True, cpp_path="gcc", cpp_args=cpp_args)

    functions = []

    class AstVisitor(c_ast.NodeVisitor):
        def visit_Decl(self, node):
            if not node.name.startswith("mts_"):
                return

            functions.append(node.name)

    visitor = AstVisitor()
    visitor.visit(ast)

    return functions


if __name__ == "__main__":
    docs = documented_functions()
    outline = functions_in_outline()
    for function in all_functions():
        if function not in docs:
            error("Missing documentation for {}".format(function))
        if function not in outline:
            error("Missing outline for {}".format(function))

    if ERRORS != 0:
        sys.exit(1)
