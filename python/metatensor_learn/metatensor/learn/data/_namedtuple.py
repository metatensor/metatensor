"""
This contains a custom version of `collections.namedtuple` that supports field names
which are not valid Python identifiers.
"""

import keyword
import sys
from operator import itemgetter


def _tuplegetter(index, doc):
    return property(itemgetter(index), doc=doc)


def namedtuple(typename, field_names):
    field_names = list(map(str, field_names))
    typename = sys.intern(str(typename))

    if keyword.iskeyword(typename):
        raise ValueError(f"Type names cannot be a keyword: {typename!r}")

    arg_names = []
    fmt_names = []
    for index, name in enumerate(field_names):
        if name.isidentifier() and not keyword.iskeyword(name):
            arg_names.append(name)
            fmt_names.append(name)
        else:
            arg_names.append(f"_{index}")
            fmt_names.append(f"'{name}'")

    seen = set()
    for name in field_names:
        if name in seen:
            raise ValueError(f"Encountered duplicate field name: {name!r}")
        seen.add(name)

    # Variables used in the methods and docstrings
    field_names = tuple(map(sys.intern, field_names))
    num_fields = len(field_names)
    arg_list = ", ".join(arg_names)
    if num_fields == 1:
        arg_list += ","
    repr_fmt = "(" + ", ".join(f"{name}=%r" for name in fmt_names) + ")"
    tuple_new = tuple.__new__
    tuple_getitem = tuple.__getitem__
    _dict, _tuple, _len, _map, _zip = dict, tuple, len, map, zip

    # Create all the named tuple methods to be added to the class namespace

    namespace = {
        "_tuple_new": tuple_new,
        "__builtins__": {},
        "__name__": f"namedtuple_{typename}",
    }
    code = f"lambda _cls, {arg_list}: _tuple_new(_cls, ({arg_list}))"
    __new__ = eval(code, namespace)
    __new__.__name__ = "__new__"
    __new__.__doc__ = f"Create new instance of {typename}({arg_list})"

    @classmethod
    def _make(cls, iterable):
        result = tuple_new(cls, iterable)
        if _len(result) != num_fields:
            raise TypeError(f"Expected {num_fields} arguments, got {len(result)}")
        return result

    _make.__func__.__doc__ = f"Make a new {typename} object from a sequence or iterable"

    def _replace(self, /, **kwds):
        result = self._make(_map(kwds.pop, field_names, self))
        if kwds:
            raise TypeError(f"Got unexpected field names: {list(kwds)!r}")
        return result

    _replace.__doc__ = (
        f"Return a new {typename} object replacing specified fields with new values"
    )

    def __repr__(self):
        "Return a nicely formatted representation string"
        return self.__class__.__name__ + repr_fmt % self

    def _asdict(self):
        "Return a new dict which maps field names to their values."
        return _dict(_zip(self._fields, self))

    def __getnewargs__(self):
        "Return self as a plain tuple.  Used by copy and pickle."
        return _tuple(self)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._asdict()[key]
        else:
            return tuple_getitem(self, key)

    # Modify function metadata to help with introspection and debugging
    for method in (
        __new__,
        _make.__func__,
        _replace,
        __repr__,
        _asdict,
        __getnewargs__,
        __getitem__,
    ):
        method.__qualname__ = f"{typename}.{method.__name__}"

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        "__doc__": f"{typename}({arg_list})",
        "__slots__": (),
        "_fields": field_names,
        "__new__": __new__,
        "_make": _make,
        "__replace__": _replace,
        "_replace": _replace,
        "__repr__": __repr__,
        "_asdict": _asdict,
        "__getnewargs__": __getnewargs__,
        "__getitem__": __getitem__,
        "__match_args__": field_names,
    }
    for index, name in enumerate(field_names):
        if name.isidentifier():
            doc = sys.intern(f"Alias for field number {index}")
            class_namespace[name] = _tuplegetter(index, doc)

    result = type(typename, (tuple,), class_namespace)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.
    try:
        module = sys._getframemodulename(1) or "__main__"
    except AttributeError:
        try:
            module = sys._getframe(1).f_globals.get("__name__", "__main__")
        except (AttributeError, ValueError):
            pass

    if module is not None:
        result.__module__ = module

    return result
