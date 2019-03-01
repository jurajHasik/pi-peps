_opt = 'mkl'
_dir = get_option(_opt+'-dir')

if _dir == '' # rely on pkg-config
    mkl = dependency('mkl-dynamic-lp64-iomp', required: true)
    # we assume the .pc file provides the correct flags
else
    _libdir = _dir+'/lib/intel64_lin'
    _omplibdir = _dir + '/../compiler/lib/intel64_lin'
    _includedir = include_directories(_dir+'/include')

    _largs = ['-L'+_libdir,
              '-lmkl_intel_lp64',
              '-lmkl_intel_thread',
              '-lmkl_core',
              '-L'+_omplibdir,
              '-liomp5',
              '-lpthread',
              '-lm',
              '-ldl']

    # check that we can actually link against mkl
    
    compiler = meson.get_compiler('cpp')

    _code = '''#include <mkl.h>
    int main(){
    }'''
    
    _compiles = compiler.compiles(_code, name:
                                  _opt.to_upper()+' compile flags check',
                                  include_directories:_includedir)


    if not _compiles
        error('\n\nCannot find suitable installation for '+_opt.to_upper())
    endif

    _libs = []
    _mkl_libs = [
              'mkl_intel_lp64',
              'mkl_intel_thread',
              'mkl_core',
              'iomp5',
              'pthread',
              'm',
              'dl']

    foreach _lib : _mkl_libs
       _libs += compiler.find_library(_lib,
                                   dirs:[_libdir,_omplibdir], required:true )
    
    endforeach
    
    mkl = declare_dependency(include_directories:
                             _includedir,
                             compile_args:'-I'+_dir+'/include',
                             dependencies:_libs)
endif

deps += mkl

