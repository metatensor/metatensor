#!/usr/bin/env python
"""
Unified declaration generator for metatensor C API bindings.

Usage:
    ./scripts/update-declarations.py            # generate all language bindings
    ./scripts/update-declarations.py python     # generate python only
    ./scripts/update-declarations.py julia      # generate julia only
    ./scripts/update-declarations.py rust       # generate rust only
"""

import os
import shutil
import subprocess
import sys
import tempfile

from pycparser import c_ast, parse_file


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
FAKE_INCLUDES = [
    os.path.join(ROOT, "python", "scripts", "include"),
    os.path.join(ROOT, "scripts", "include"),
]
METATENSOR_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatensor-core", "include", "metatensor.h")
)


# ============================================================================ #
# Shared AST parsing
# ============================================================================ #


class Function:
    def __init__(self, name, restype):
        self.name = name
        self.restype = restype
        self.args = []

    def add_arg(self, name, type):
        self.args.append((name, type))


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
        node_name = node.name
        if node_name is None:
            node_name = node.type.name

        if not node_name.startswith("mts_"):
            return

        if isinstance(node.type, c_ast.Enum):
            enum = Enum(node_name)
            for enumerator in node.type.values.enumerators:
                # Strip C unsigned/long suffixes (e.g. 0U, 1UL)
                value = enumerator.value.value.rstrip("UuLl")
                enum.add_value(enumerator.name, value)
            self.enums.append(enum)
        elif isinstance(node.type, c_ast.FuncDecl):
            function = Function(node.name, node.type.type)
            for parameter in node.type.args.params:
                function.add_arg(parameter.name, parameter.type)
            self.functions.append(function)
        else:
            raise RuntimeError(f"Unknown declaration type for {node_name}")

    def visit_Typedef(self, node):
        # Extract metatensor and dlpack stuff only
        if not (node.name.startswith("mts_") or node.name.startswith("DL")):
            return

        if isinstance(node.type.type, c_ast.Enum):
            enum = Enum(node.name)
            for enumerator in node.type.type.values.enumerators:
                # Strip C unsigned/long suffixes (e.g. 0U, 1UL)
                value = enumerator.value.value.rstrip("UuLl")
                enum.add_value(enumerator.name, value)
            self.enums.append(enum)

        elif isinstance(node.type.type, c_ast.Struct):
            if node.name.startswith("DLPackExchangeAPI"):
                return

            struct = Struct(node.name)
            for _, member in node.type.type.children():
                struct.add_member(member.name, member.type)
            self.structs.append(struct)

        else:
            self.types[node.name] = node.type.type


def _typedecl_name(type):
    assert isinstance(type, c_ast.TypeDecl)
    if isinstance(type.type, c_ast.Struct):
        return type.type.name
    elif isinstance(type.type, c_ast.Enum):
        return type.type.name
    else:
        assert len(type.type.names) == 1
        return type.type.names[0]


def parse_header(file):
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


# ==================================================================================== #
#                                 Python backend                                       #
# ==================================================================================== #


def _py_type_name(name):
    if name.startswith("mts_") or name.startswith("DL"):
        return name
    elif name == "uintptr_t":
        return "c_uintptr_t"
    elif name == "void":
        return "None"
    elif name == "int8_t":
        return "ctypes.c_int8"
    elif name == "uint8_t":
        return "ctypes.c_uint8"
    elif name == "int16_t":
        return "ctypes.c_int16"
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


def _py_funcdecl(type):
    restype = _py_type(type.type)
    args = [_py_type(t.type) for t in type.args.params]
    return f"CFUNCTYPE({restype}, {', '.join(args)})"


def _py_type(type):
    if isinstance(type, c_ast.PtrDecl):
        if isinstance(type.type, c_ast.PtrDecl):
            if isinstance(type.type.type, c_ast.TypeDecl):
                name = _typedecl_name(type.type.type)
                if name == "char":
                    return "POINTER(ctypes.c_char_p)"
                elif name == "uint8_t":
                    return "POINTER(ctypes.c_char_p)"
                name = _py_type_name(name)
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
                return f"POINTER({_py_type_name(name)})"
        elif isinstance(type.type, c_ast.FuncDecl):
            return _py_funcdecl(type.type)
    else:
        if isinstance(type, c_ast.TypeDecl):
            return _py_type_name(_typedecl_name(type))
        elif isinstance(type, c_ast.IdentifierType):
            return _py_type_name(type.names[0])
        elif isinstance(type, c_ast.ArrayDecl):
            if isinstance(type.dim, c_ast.Constant):
                size = type.dim.value
            else:
                raise Exception("dynamically sized arrays are not supported")
            return f"{_py_type(type.type)} * {size}"
        elif isinstance(type, c_ast.FuncDecl):
            return _py_funcdecl(type)
    raise Exception("Unknown type")


def generate_python(data):
    outpath = os.path.join(ROOT, "python", "metatensor_core", "metatensor", "_c_api.py")
    with open(outpath, "w") as f:
        f.write(
            """# fmt: off
# flake8: noqa
\"\"\"
This file declares the C-API corresponding to metatensor.h, in a way compatible
with the ctypes Python module.

This file is automatically generated by `scripts/update-declarations.py`,
do not edit it manually!
\"\"\"

import ctypes
import platform
from ctypes import CFUNCTYPE, POINTER


class EnumType(type(ctypes.c_int32)):
    def __new__(metacls, name, bases, dict):
        if "_members_" not in dict:
            _members_ = {}
            for key, value in dict.items():
                if not key.startswith("_"):
                    _members_[key] = value

            dict["_members_"] = _members_
        else:
            _members_ = dict["_members_"]

        dict["_reverse_map_"] = {v: k for k, v in _members_.items()}
        cls = type(ctypes.c_int32).__new__(metacls, name, bases, dict)
        for key, value in cls._members_.items():
            globals()[key] = value
        return cls

    def __repr__(self):
        return "<Enumeration %s>" % self.__name__


class Enum(ctypes.c_int32):
    __metaclass__ = EnumType
    _members_ = {}

    def __repr__(self):
        value_str = self._reverse_map_.get(self.value, str(self.value))
        return f"{self.__class__.__name__}.{value_str}"

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other

        return type(self) is type(other) and self.value == other.value


arch = platform.architecture()[0]
if arch == "32bit":
    c_uintptr_t = ctypes.c_uint32
elif arch == "64bit":
    c_uintptr_t = ctypes.c_uint64

"""
        )

        # Enums
        for enum in data.enums:
            f.write(f"\n\nclass {enum.name}(Enum):\n")
            for name, value in enum.values.items():
                f.write(f"    {name} = {value}\n")

        # structs declartions, without fields
        for struct in data.structs:
            f.write(f"\n\nclass {struct.name}(ctypes.Structure):\n")
            f.write("    pass\n")

        # typedefs
        f.write("\n\n")
        for name, c_type in data.types.items():
            if name == "mts_status_t":
                # this is already defined as an enum
                continue
            f.write(f"{name} = {_py_type(c_type)}\n")

        # structs fields definitions
        f.write("\n")
        for struct in data.structs:
            if len(struct.members) == 0:
                continue
            f.write(f"\n{struct.name}._fields_ = [\n")
            for name, type in struct.members.items():
                f.write(f'    ("{name}", {_py_type(type)}),\n')
            f.write("]\n")

        # Functions
        f.write("\n\ndef setup_functions(lib):\n")
        f.write("    from .status import _check_status\n")
        for function in data.functions:
            f.write(f"\n    lib.{function.name}.argtypes = [")
            args = [_py_type(arg[1]) for arg in function.args]
            if args == ["None"]:
                args = []
            for arg in args:
                f.write(f"\n        {arg},")
            f.write("\n    ]\n")
            restype = _py_type(function.restype)
            if restype == "mts_status_t" and function.name != "mts_last_error":
                restype = "_check_status"
            f.write(f"    lib.{function.name}.restype = {restype}\n")


# ==================================================================================== #
#                                 Julia backend                                        #
# ==================================================================================== #

CTYPES_TO_JULIA = {
    "uintptr_t": "UIntptr",
    "int8_t": "Int8",
    "uint8_t": "UInt8",
    "int16_t": "Int16",
    "uint16_t": "UInt16",
    "int32_t": "Int32",
    "uint32_t": "UInt32",
    "int64_t": "Int64",
    "uint64_t": "UInt64",
}


def _jl_type_name(name):
    if name.startswith("mts_") or name.startswith("DL"):
        return name
    if name in CTYPES_TO_JULIA:
        return CTYPES_TO_JULIA[name]
    else:
        return "C" + name


def _jl_funcdecl(type):
    restype = _jl_type(type.type)
    args = [_jl_type(t.type) for t in type.args.params]
    return f"Ptr{{Cvoid}} #= ({', '.join(args)}) -> {restype} =#"


def _jl_type(type):
    if isinstance(type, c_ast.PtrDecl):
        if isinstance(type.type, c_ast.PtrDecl):
            if isinstance(type.type.type, c_ast.TypeDecl):
                name = _typedecl_name(type.type.type)
                return f"Ptr{{Ptr{{{_jl_type_name(name)}}}}}"
            elif isinstance(type.type.type, c_ast.PtrDecl):
                assert isinstance(type.type.type.type, c_ast.TypeDecl)
                assert _typedecl_name(type.type.type.type) == "char"
                return f"Ptr{{Ptr{{Ptr{{{_jl_type_name('char')}}}}}}}"
        elif isinstance(type.type, c_ast.TypeDecl):
            name = _typedecl_name(type.type)
            return f"Ptr{{{_jl_type_name(name)}}}"
        elif isinstance(type.type, c_ast.FuncDecl):
            return _jl_funcdecl(type.type)
    else:
        if isinstance(type, c_ast.TypeDecl):
            return _jl_type_name(_typedecl_name(type))
        elif isinstance(type, c_ast.IdentifierType):
            return _jl_type_name(type.names[0])
        elif isinstance(type, c_ast.ArrayDecl):
            if isinstance(type.dim, c_ast.Constant):
                size = type.dim.value
            else:
                raise Exception("dynamically sized arrays are not supported")
            return f"{_jl_type(type.type)} * {size}"
        elif isinstance(type, c_ast.FuncDecl):
            return _jl_funcdecl(type)
    raise Exception("Unknown type")


def generate_julia(data):
    outpath = os.path.join(ROOT, "julia", "src", "generated", "_c_api.jl")
    with open(outpath, "w") as f:
        f.write(
            """# This file declares the C-API corresponding to metatensor.h

# This file is automatically generated by `scripts/update-declarations.py`,
# do not edit it manually!


# ========== Manual definitions ========= #
if Sys.WORD_SIZE == 32
    UIntptr = UInt32
elseif Sys.WORD_SIZE == 64
    UIntptr = UInt64
else
    error("unknown size of uintptr_t")
end

Cbool = Cuchar
mts_data_origin_t = UInt64

mts_create_array_callback_t = Ptr{Cvoid}  # TODO: actual type
mts_realloc_buffer_t = Ptr{Cvoid}         # TODO: actual type

# ====== End of manual definitions ====== #
"""
        )

        # Enums
        f.write("\n\n# ===== Enum definitions\n")
        for enum in data.enums:
            f.write(f"\n\n# enum {enum.name}\nconst {enum.name} = UInt32\n")
            for name, value in enum.values.items():
                f.write(f"const {name} = {enum.name}({value})\n")

        # mts_* structs
        f.write("\n\n# ===== Struct definitions\n")
        for struct in data.structs:
            f.write(f"struct {struct.name}\n")
            for name, type in struct.members.items():
                f.write(f"    {name} :: {_jl_type(type)}\n")
            f.write("end\n\n")

        # Functions
        f.write("\n# ===== Function definitions\n")
        for function in data.functions:
            args = [(arg[0], _jl_type(arg[1])) for arg in function.args]
            if args == [(None, "Cvoid")]:
                args = []
            names_types = [f"{a[0]}::{a[1]}" for a in args]
            if len(args) == 0:
                types_tuple = "()"
            else:
                types_tuple = f"({', '.join([a[1] for a in args])},)"
            restype = _jl_type(function.restype)
            f.write(f"\nfunction {function.name}({', '.join(names_types)})\n")
            f.write(f"    ccall((:{function.name}, libmetatensor), \n")
            f.write(f"        {restype},\n")
            f.write(f"        {types_tuple},\n")
            f.write(f"        {', '.join([a[0] for a in args])}\n    )\n")
            f.write("end\n")


# ==================================================================================== #
#                                 Rust backend                                         #
# ==================================================================================== #

RUST_START = """
#![allow(warnings)]
//! Rust definition corresponding to metatensor-core C-API.
//!
//! This module is exported for advanced users of the metatensor crate, but
//! should not be needed by most.

use dlpk::sys::*;

#[cfg_attr(feature="static", link(name="metatensor", kind = "static", modifiers = "-whole-archive"))]
#[cfg_attr(all(not(feature="static"), not(target_os="windows")), link(name="metatensor", kind = "dylib"))]
#[cfg_attr(all(not(feature="static"), target_os="windows"), link(name="metatensor.dll", kind = "dylib"))]
extern "C" {}
"""  # noqa: E501


def generate_rust():
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "bindgen",
                METATENSOR_HEADER,
                "--output",
                os.path.join(tmpdir, "c_api.rs"),
                "--disable-header-comment",
                "--no-doc-comments",
                "--merge-extern-blocks",
                "--must-use-type",
                "mts_status_t",
                "--allowlist-function",
                "^mts_.*",
                "--allowlist-type",
                "^mts_.*",
                "--allowlist-var",
                "^MTS_.*",
                "--no-prepend-enum-name",
                "--blocklist-type",
                "DL.*",
                "--rust-target",
                "1.74",
                "--raw-line",
                RUST_START,
                "--",
                "-I",
                os.path.join(ROOT, "scripts", "include"),
            ],
            check=True,
        )

        subprocess.run(
            ["rustfmt", os.path.join(tmpdir, "c_api.rs")],
            check=True,
            cwd=tmpdir,
        )

        shutil.copyfile(
            os.path.join(tmpdir, "c_api.rs"),
            os.path.join(ROOT, "rust", "metatensor-sys", "src", "c_api.rs"),
        )


# ==================================================================================== #
#                                       main                                           #
# ==================================================================================== #


def main():
    data = parse_header(METATENSOR_HEADER)

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["python", "julia", "rust"]

    for target in targets:
        if target == "python":
            generate_python(data)
        elif target == "julia":
            generate_julia(data)
        elif target == "rust":
            generate_rust()
        else:
            print(f"Unknown target: {target}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
