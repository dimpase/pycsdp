#!/usr/bin/env python
from os.path import join
import os
SAGE_LIB = os.environ['SAGE_LOCAL']+'/lib'
SAGE_INCLUDE = os.environ['SAGE_LOCAL']+'/include'

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('pycsdp', parent_package, top_path)
    ##########################################################################
    # Adjust the libname (so libsdp.so/a <-> sdp), the library path and 
    # provide the location of the "declarations.h" file.
    ##########################################################################
    libname = 'sdp'
    library_dir = [SAGE_LIB]
    includes = [SAGE_INCLUDE, '.']

    sources = [join('Src/','*.cxx'),
               join('Src/','*.c'),
               '_csdp.cpp']
    lapack_opt = get_info('lapack_opt')

    config.add_extension('_csdp',
                         sources = sources,
                         include_dirs = includes,
                         library_dirs = library_dir,
                         extra_compile_args = ['-funroll-loops'],
                         define_macros = [('NOSHORTS',None)],
                         libraries = [libname],
                         extra_info = lapack_opt)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration, version='0.1.0')
