#!/usr/bin/env python
import os

from pycparser import c_ast, parse_file


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAKE_INCLUDES = [
    os.path.join(ROOT, "python", "scripts", "include"),
    os.path.join(ROOT, "scripts", "include"),
]
VENDORED_INCLUDES = os.path.join(ROOT, "metatensor-core", "include", "dlpack")
METATENSOR_HEADER = os.path.relpath(
    os.path.join(ROOT, "metatensor-core", "include", "metatensor.h")
)

DLPACK_TYPES = {
    "DLManagedTensorVersioned",
}


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
                function.add_arg(parameter.name, parameter.type)
        self.functions.append(function)

    def visit_Typedef(self, node):
        if not node.name.startswith("mts_") and node.name not in DLPACK_TYPES:
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
            # check if we already have this struct from a typedef
            if any(s.name == node.name for s in self.structs):
                return

            struct = Struct(node.name)
            for member in node.decls:
                struct.add_member(member.name, member.type)
            self.structs.append(struct)


def parse(file):
    cpp_args = ["-E"]
    for path in FAKE_INCLUDES:
        cpp_args += ["-I", path]
    cpp_args += ["-I", VENDORED_INCLUDES]
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


CTYPES_TO_JULIA = {
    "uintptr_t": "UIntptr",
    "uint8_t": "UInt8",
    "uint16_t": "UInt16",
    "int32_t": "Int32",
    "uint32_t": "UInt32",
    "int64_t": "Int64",
    "uint64_t": "UInt64",
}


def c_type_name(name):
    if name.startswith("mts_"):
        return name
    elif name in DLPACK_TYPES:
        return f"C{name}"
    elif name in CTYPES_TO_JULIA:
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
    args = []
    if type.args:
        args = [type_to_julia(t.type) for t in type.args.params]

    return f"Ptr{{Cvoid}} #= ({', '.join(args)}) -> {restype} =#"


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

    raise Exception(f"Unknown type: {type.__class__}")


def generate_enums(file, enums):
    for enum in enums:
        # a bit of a hack to not generate julia enums for C enums
        if not enum.name.startswith("mts_"):
            for name, value in enum.values.items():
                file.write(f"const {name} = {value}\n")
        else:
            file.write(f"\n\n# enum {enum.name}\nconst {enum.name} = UInt32\n")
            for name, value in enum.values.items():
                file.write(f"const {name} = {enum.name}({value})\n")


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
        file.write(f"struct {c_type_name(struct.name)}\n")
        if len(struct.members) == 0:
            file.write("    # opaque struct\n")
        else:
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
        if len(args) > 0:
            file.write(f"        {', '.join([a[0] for a in args])}\n    )\n")
        else:
            file.write("    )\n")
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
mts_realloc_buffer_t = Ptr{Cvoid}      # TODO: actual type

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
