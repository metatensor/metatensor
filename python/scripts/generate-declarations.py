#!/usr/bin/env python
import os

from pycparser import c_ast, parse_file


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAKE_INCLUDES = [
    os.path.join(ROOT, "python", "scripts", "include"),
    os.path.join(ROOT, "scripts", "include"),
]
METATENSOR_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatensor-core", "include", "metatensor.h")
)


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
        if not node.name.startswith("mts_"):
            return

        function = Function(node.name, node.type.type)
        for parameter in node.type.args.params:
            function.add_arg(parameter.type)
        self.functions.append(function)

    def visit_Typedef(self, node):
        if not node.name.startswith("mts_"):
            return

        if isinstance(node.type.type, c_ast.Enum):
            # Get name and value for enum
            enum = Enum(node.name)
            for enumerator in node.type.type.values.enumerators:
                enum.add_value(enumerator.name, enumerator.value.value)
            self.enums.append(enum)

        elif isinstance(node.type.type, c_ast.Struct):
            struct = Struct(node.name)
            for _, member in node.type.type.children():
                struct.add_member(member.name, member.type)

            self.structs.append(struct)

        else:
            self.types[node.name] = node.type.type


def parse(file):
    cpp_args = ["-E"]
    for path in FAKE_INCLUDES:
        cpp_args += ["-I", path]
    ast = parse_file(file, use_cpp=True, cpp_path="gcc", cpp_args=cpp_args)

    visitor = AstVisitor()
    visitor.visit(ast)

    with open(file) as fd:
        for line in fd:
            if "#define" in line:
                split = line.split()
                name = split[1]
                if name == "METATENSOR_H":
                    continue
                value = split[2]

                visitor.defines[name] = value
    return visitor


def c_type_name(name):
    if name.startswith("mts_"):
        return name
    elif name == "uintptr_t":
        return "c_uintptr_t"
    elif name == "void":
        return "None"
    elif name == "uint8_t":
        return "ctypes.c_uint8"
    elif name == "int32_t":
        return "ctypes.c_int32"
    elif name == "uint32_t":
        return "ctypes.c_uint32"
    elif name == "int64_t":
        return "ctypes.c_int64"
    elif name == "uint64_t":
        return "ctypes.c_uint64"
    elif name == "DLDevice":
        return "DLDevice"
    elif name == "DLPackVersion":
        return "DLPackVersion"
    elif name == "DLManagedTensorVersioned":
        return "DLManagedTensorVersioned"
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

    raise Exception("Unknown type")


def generate_enums(file, enums):
    for enum in enums:
        file.write(f"\n\nclass {enum.name}(enum.Enum):\n")
        for name, value in enum.values.items():
            file.write(f"    {name} = {value}\n")


def generate_structs(file, structs):
    for struct in structs:
        file.write(f"\n\nclass {struct.name}(ctypes.Structure):\n")
        file.write("    pass\n")

        if len(struct.members) == 0:
            continue

        file.write(f"\n{struct.name}._fields_ = [\n")
        for name, type in struct.members.items():
            file.write(f'    ("{name}", {type_to_ctypes(type)}),\n')
        file.write("]\n")


def generate_functions(file, functions):
    file.write("\n\ndef setup_functions(lib):\n")
    file.write("    from .status import _check_status\n")

    for function in functions:
        file.write(f"\n    lib.{function.name}.argtypes = [")
        args = [type_to_ctypes(arg) for arg in function.args]

        # functions taking void parameter in C don't have any parameter
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
            """# fmt: off
# flake8: noqa
\"\"\"
This file declares the C-API corresponding to metatensor.h, in a way compatible
with the ctypes Python module.

This file is automatically generated by `python/scripts/generate-declarations.py`,
do not edit it manually!
\"\"\"

import ctypes
import enum
import platform
from ctypes import CFUNCTYPE, POINTER


arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

"""
        )
        for name, value in data.defines.items():
            file.write(f"{name} = {value}\n")
        file.write("\n\n")

        for name, c_type in data.types.items():
            if name == "mts_create_array_callback_t":
                # will be generated below, it depends on the structs
                continue
            file.write(f"{name} = {type_to_ctypes(c_type)}\n")

        # --- Manual definitions for the DLPack structs (ala pydlpack) ---
        file.write("""
# ============================================================================ #
# DLPack types
# ============================================================================ #
class DLPackVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]

class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]

class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", POINTER(ctypes.c_int64)),
        ("strides", POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

class DLManagedTensorVersioned(ctypes.Structure):
    pass

_DLManagedTensorVersionedDeleter = CFUNCTYPE(None, POINTER(DLManagedTensorVersioned))

DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorVersionedDeleter),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", DLTensor),
]
""")
        # -----------------------------------------------------

        generate_enums(file, data.enums)
        generate_structs(file, data.structs)

        file.write("\n\n")
        callback_type = type_to_ctypes(data.types["mts_create_array_callback_t"])
        file.write(f"mts_create_array_callback_t = {callback_type}\n")

        generate_functions(file, data.functions)


if __name__ == "__main__":
    generate_declarations()
