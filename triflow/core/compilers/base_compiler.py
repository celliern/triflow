import warnings
import attr
from fuzzywuzzy import process

from ..grid_builder import GridBuilder
from ..system import PDESys

available_compilers = {}

def get_compiler(name):
    """get a compiler by its name

    Arguments:
        name {str} -- compiler name

    Raises:
        NotImplementedError -- raised if the compiler is not available.

    Returns:
        Compiler -- the requested compiler
    """
    try:
        return available_compilers[name]
    except KeyError:
        err_msg = "%s compiler is not registered." % name
        (suggest, score), = process.extract(name,
                                            available_compilers.keys(),
                                            limit=1)
        if score > 70:
            err_msg += ("\n%s is available and seems to be close. "
                        "It may be what you are looking for !" % suggest)
        err_msg += ("\nFull list of available compilers:\n\t- %s" %
                    ("\n\t- ".join(available_compilers.keys())))
        raise NotImplementedError(err_msg)

def register_compiler(CustomCompiler):
    global available_compilers
    if Compiler not in CustomCompiler.__mro__:
        raise AttributeError("The provider compiler should inherit from the "
                             "Compiler base class.")
    available_compilers[CustomCompiler.name] = CustomCompiler

@attr.s
class Compiler:
    system = attr.ib(type=PDESys)
    grid_builder = attr.ib(type=GridBuilder)