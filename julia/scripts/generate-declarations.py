#!/usr/bin/env python
import os

from pycparser import c_ast, parse_file


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FAKE_INCLUDES = os.path.join(ROOT, "python", "scripts", "include")
VENDORED_INCLUDES = os.path.join(ROOT, "metatensor-core", "include", "vendored")
METATENSOR_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatensor-core", "include", "metatensor.h")
)


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
        if not node.name or not node.name.startswith("mts_"):
            return

        function = Function(node.name, node.type.type)
        for parameter in node.type.args.params:
            function.add_arg(parameter.name, parameter.type)
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
    cpp_args = ["-E", f"-I{FAKE_INCLUDES}", f"-I{VENDORED_INCLUDES}"]
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


CTYPES_TO_JULIA = {
    "uintptr_t": "UIntptr",
    "uint8_t": "UInt8",
    "int32_t": "Int32",
    "uint32_t": "UInt32",
    "int64_t": "Int64",
    "uint64_t": "UInt64",
}


def c_type_name(name):
    if name.startswith("mts_") or name == "ManagedTensorVersioned":
        return name
    if name in CTYPES_TO_JULIA:
        return CTYPES_TO_JULIA[name]
    else:
        return "C" + name


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
    restype = type_to_julia(type.type)
    args = [type_to_julia(t.type) for t in type.args.params]

    return f'Ptr{{Cvoid}} #= ({", ".join(args)}) -> {restype} =#'


def type_to_julia(type):
    if isinstance(type, c_ast.PtrDecl):
        if isinstance(type.type, c_ast.PtrDecl):
            if isinstance(type.type.type, c_ast.TypeDecl):
                name = _typedecl_name(type.type.type)
                return f"Ptr{{Ptr{{{c_type_name(name)}}}}}"
            elif isinstance(type.type.type, c_ast.PtrDecl):
                assert isinstance(type.type.type.type, c_ast.TypeDecl)
                assert _typedecl_name(type.type.type.type) == "char"
                return f"Ptr{{Ptr{{Ptr{{{c_type_name('char')}}}}}}}"

        elif isinstance(type.type, c_ast.TypeDecl):
            name = _typedecl_name(type.type)
            return f"Ptr{{{c_type_name(name)}}}"

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

            return f"{type_to_julia(type.type)} * {size}"
        elif isinstance(type, c_ast.FuncDecl):
            return funcdecl_to_ctypes(type)

    raise Exception("Unknown type")


def generate_enums(file, enums):
    for enum in enums:
        file.write(f"\n\n# enum {enum.name}\nconst {enum.name} = UInt32\n")
        for name, value in enum.values.items():
            file.write(f"const {name} = {enum.name}({value})\n")


def generate_structs(file, structs):
    for struct in structs:
        file.write(f"struct {struct.name}\n")
        for name, type in struct.members.items():
            file.write(f"    {name} :: {type_to_julia(type)}\n")
        file.write("end\n\n")


def generate_functions(file, functions):
    for function in functions:
        args = [(arg[0], type_to_julia(arg[1])) for arg in function.args]

        if args == [(None, "Cvoid")]:
            # void function in the C API, no actual parameters
            args = []

        names_types = [f"{a[0]}::{a[1]}" for a in args]

        if len(args) == 0:
            types_tuple = "()"
        else:
            types_tuple = f"({', '.join([a[1] for a in args])},)"

        restype = type_to_julia(function.restype)

        file.write(f"\nfunction {function.name}({', '.join(names_types)})\n")
        file.write(f"    ccall((:{function.name}, libmetatensor), \n")
        file.write(f"        {restype},\n")
        file.write(f"        {types_tuple},\n")
        file.write(f"        {', '.join([a[0] for a in args])}\n    )\n")
        file.write("end\n")


def generate_declarations():
    data = parse(METATENSOR_HEADER)

    outpath = os.path.join(
        ROOT,
        "julia",
        "src",
        "generated",
        "_c_api.jl",
    )

    with open(outpath, "w") as file:
        file.write(
            """# This file declares the C-API corresponding to metatensor.h

# This file is automatically generated by `julia/scripts/generate-declarations.py`,
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
mts_status_t = Int32
mts_data_origin_t = UInt64

mts_create_array_callback_t = Ptr{Cvoid}  # TODO: actual type
mts_realloc_buffer_t = Ptr{Cvoid}         # TODO: actual type

# ====== Enf of manual definitions ====== #
"""
        )

        file.write("\n\n# ===== Macros definitions\n")
        for name, value in data.defines.items():
            file.write(f"{name} = {value}\n")

        file.write("\n\n# ===== Enum definitions\n")
        generate_enums(file, data.enums)

        file.write("\n\n# ===== Struct definitions\n")
        generate_structs(file, data.structs)

        file.write("\n\n# ===== Function definitions\n")
        generate_functions(file, data.functions)


if __name__ == "__main__":
    generate_declarations()
