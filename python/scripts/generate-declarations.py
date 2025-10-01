#!/usr/bin/env python
import os

from pycparser import c_ast, parse_file


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FAKE_INCLUDES = os.path.join(ROOT, "python", "scripts", "include")
VENDORED_INCLUDES = os.path.join(ROOT, "metatensor-core", "include", "vendored")
METATENSOR_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatensor-core", "include", "metatensor.h")
)

DLPACK_TYPES = {
    "DLPackVersion",
    "DLDevice",
    "DLDataType",
    "DLTensor",
    "DLManagedTensorVersioned",
}

DLPACK_ENUMS = {
    "DLDeviceType",
}


class Function:
    def __init__(self, name, restype):
        self.name = name
        self.restype = restype
        self.args = []

    def add_arg(self, arg):
        self.args.append(arg)


class Struct:
    def __init__(self, name):
        self.name = name
        self.members = {}

    def add_member(self, name, type):
        self.members[name] = type


class Enum:
    def __init__(self, name):
        self.name = name
        self.values = {}

    def add_value(self, name, value):
        self.values[name] = value


class AstVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.enums = []
        self.structs = []
        self.types = {}
        self.defines = {}

    def visit_Decl(self, node):
        if not node.name or not node.name.startswith("mts_"):
            return

        # only visit function declarations
        if not isinstance(node.type, c_ast.FuncDecl):
            return

        function = Function(node.name, node.type.type)
        if node.type.args:
            for parameter in node.type.args.params:
                # C functions with no arguments have a single `void` parameter
                if isinstance(parameter.type, c_ast.TypeDecl) and isinstance(
                    parameter.type.type, c_ast.IdentifierType
                ):
                    if parameter.type.type.names[0] == "void":
                        continue
                function.add_arg(parameter.type)
        self.functions.append(function)

    def visit_Typedef(self, node):
        if (
            not node.name.startswith("mts_")
            and node.name not in DLPACK_TYPES
            and node.name not in DLPACK_ENUMS
        ):
            return

        if isinstance(node.type.type, c_ast.Enum):
            enum = Enum(node.name)
            if node.type.type.values:
                for enumerator in node.type.type.values.enumerators:
                    enum.add_value(enumerator.name, enumerator.value.value)
            self.enums.append(enum)

        elif isinstance(node.type.type, c_ast.Struct):
            # handle `typedef struct { ... } name;`
            struct = Struct(node.name)

            if node.type.type.decls is not None:
                for member in node.type.type.decls:
                    struct.add_member(member.name, member.type)

            self.structs.append(struct)

        else:
            self.types[node.name] = node.type.type

    def visit_Struct(self, node):
        if node.name in DLPACK_TYPES and node.decls:
            # check if this struct has already been covered by a typedef
            if any(s.name == node.name for s in self.structs):
                return

            struct = Struct(node.name)
            for member in node.decls:
                struct.add_member(member.name, member.type)
            self.structs.append(struct)


def parse(file):
    cpp_args = ["-E", f"-I{FAKE_INCLUDES}", f"-I{VENDORED_INCLUDES}"]
    ast = parse_file(file, use_cpp=True, cpp_path="gcc", cpp_args=cpp_args)

    visitor = AstVisitor()
    visitor.visit(ast)

    with open(file) as fd:
        for line in fd:
            if "#define" in line:
                split = line.split()
                if len(split) < 3:
                    continue

                name = split[1]
                if name == "METATENSOR_H" or not (
                    name.startswith("MTS_") or name.startswith("DLPACK_")
                ):
                    continue
                value = split[2]

                visitor.defines[name] = value
    return visitor


def c_type_name(name):
    if name in DLPACK_ENUMS:
        return "ctypes.c_int32"
    # add a 'c_' prefix to DLPack types, but not to mts_ types
    elif name in DLPACK_TYPES:
        return f"c_{name}"
    elif name.startswith("mts_"):
        return name
    elif name == "uintptr_t":
        return "c_uintptr_t"
    elif name == "void":
        return "None"
    elif name == "uint8_t":
        return "ctypes.c_uint8"
    elif name == "uint16_t":
        return "ctypes.c_uint16"
    elif name == "int32_t":
        return "ctypes.c_int32"
    elif name == "uint32_t":
        return "ctypes.c_uint32"
    elif name == "int64_t":
        return "ctypes.c_int64"
    elif name == "uint64_t":
        return "ctypes.c_uint64"
    else:
        return "ctypes.c_" + name


def _typedecl_name(type):
    assert isinstance(type, c_ast.TypeDecl)
    if isinstance(type.type, c_ast.Struct):
        return type.type.name
    elif isinstance(type.type, c_ast.Enum):
        return type.type.name
    else:
        assert len(type.type.names) == 1
        return type.type.names[0]


def funcdecl_to_ctypes(type):
    restype = type_to_ctypes(type.type)
    args = []
    if type.args:
        args = [type_to_ctypes(t.type) for t in type.args.params]

    return f"CFUNCTYPE({restype}, {', '.join(args)})"


def type_to_ctypes(type):
    if isinstance(type, c_ast.PtrDecl):
        if isinstance(type.type, c_ast.PtrDecl):
            if isinstance(type.type.type, c_ast.TypeDecl):
                name = _typedecl_name(type.type.type)
                if name == "char":
                    return "POINTER(ctypes.c_char_p)"
                elif name == "uint8_t":
                    return "POINTER(ctypes.c_char_p)"

                name = c_type_name(name)
                return f"POINTER(POINTER({name}))"
            elif isinstance(type.type.type, c_ast.PtrDecl):
                assert isinstance(type.type.type.type, c_ast.TypeDecl)
                assert _typedecl_name(type.type.type.type) == "char"
                return "POINTER(POINTER(ctypes.c_char_p))"

        elif isinstance(type.type, c_ast.TypeDecl):
            name = _typedecl_name(type.type)
            if name == "void":
                return "ctypes.c_void_p"
            elif name == "char":
                return "ctypes.c_char_p"
            elif name == "uint8_t":
                return "ctypes.c_char_p"
            else:
                return f"POINTER({c_type_name(name)})"

        elif isinstance(type.type, c_ast.FuncDecl):
            return funcdecl_to_ctypes(type.type)

    else:
        # not a pointer
        if isinstance(type, c_ast.TypeDecl):
            return c_type_name(_typedecl_name(type))
        elif isinstance(type, c_ast.IdentifierType):
            return c_type_name(type.names[0])
        elif isinstance(type, c_ast.ArrayDecl):
            if isinstance(type.dim, c_ast.Constant):
                size = type.dim.value
            else:
                raise Exception("dynamically sized arrays are not supported")

            return f"{type_to_ctypes(type.type)} * {size}"
        elif isinstance(type, c_ast.FuncDecl):
            return funcdecl_to_ctypes(type)

    raise Exception(f"Unknown type: {type.__class__}")


def generate_enums(file, enums):
    for enum in enums:
        # a bit of a hack to not generate python enums for C enums
        if not enum.name.startswith("mts_"):
            for name, value in enum.values.items():
                file.write(f"{name} = {value}\n")
        else:
            file.write(f"\n\nclass {c_type_name(enum.name)}(enum.Enum):\n")
            for name, value in enum.values.items():
                file.write(f"    {name} = {value}\n")


def generate_structs(file, structs):
    # sort structs to have dependencies defined before users
    sorted_structs = []
    names = [s.name for s in structs]
    while len(sorted_structs) != len(structs):
        newly_added = 0
        for struct in structs:
            if struct in sorted_structs:
                continue

            can_be_added = True
            for member in struct.members.values():
                # find the base type of the member
                base_type = member
                while isinstance(base_type, c_ast.PtrDecl):
                    base_type = base_type.type

                if isinstance(base_type, c_ast.TypeDecl) and isinstance(
                    base_type.type, c_ast.Struct
                ):
                    member_name = base_type.type.name
                    if member_name in names and member_name != struct.name:
                        # is the dependency already in sorted_structs?
                        if not any(s.name == member_name for s in sorted_structs):
                            can_be_added = False
                            break

            if can_be_added:
                sorted_structs.append(struct)
                newly_added += 1

        if newly_added == 0:
            raise Exception("cyclic dependency in structs")

    for struct in sorted_structs:
        file.write(f"\n\nclass {c_type_name(struct.name)}(ctypes.Structure):\n")
        file.write("    pass\n")

        if len(struct.members) == 0:
            continue

        file.write(f"\n{c_type_name(struct.name)}._fields_ = [\n")
        for name, type in struct.members.items():
            file.write(f'    ("{name}", {type_to_ctypes(type)}),\n')
        file.write("]\n")


def generate_functions(file, functions):
    file.write("\n\ndef setup_functions(lib):\n")
    file.write("    from .status import _check_status\n")

    for function in functions:
        file.write(f"\n    lib.{function.name}.argtypes = [")
        args = [type_to_ctypes(arg) for arg in function.args]

        if args == ["None"]:
            args = []

        for arg in args:
            file.write(f"\n        {arg},")
        file.write("\n    ]\n")

        restype = type_to_ctypes(function.restype)
        if restype == "mts_status_t":
            restype = "_check_status"

        file.write(f"    lib.{function.name}.restype = {restype}\n")


def generate_declarations():
    data = parse(METATENSOR_HEADER)

    outpath = os.path.join(
        ROOT,
        "python",
        "metatensor_core",
        "metatensor",
        "_c_api.py",
    )
    with open(outpath, "w") as file:
        file.write(
            '''# fmt: off
# flake8: noqa
"""
This file declares the C-API corresponding to metatensor.h, in a way compatible
with the ctypes Python module.

This file is automatically generated by `python/scripts/generate-declarations.py`,
do not edit it manually!
"""

import ctypes
import enum
import platform
from ctypes import CFUNCTYPE, POINTER


arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

'''
        )
        for name, value in data.defines.items():
            file.write(f"{name} = {value}\n")
        file.write("\n\n")

        for name, c_type in data.types.items():
            if name == "mts_create_array_callback_t":
                continue
            file.write(f"{c_type_name(name)} = {type_to_ctypes(c_type)}\n")

        generate_enums(file, data.enums)
        generate_structs(file, data.structs)

        file.write("\n\n")
        callback_type = type_to_ctypes(data.types["mts_create_array_callback_t"])
        file.write(f"{c_type_name('mts_create_array_callback_t')} = {callback_type}\n")

        generate_functions(file, data.functions)


if __name__ == "__main__":
    generate_declarations()
