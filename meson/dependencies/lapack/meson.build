# lapack

compiler = meson.get_compiler('cpp')
_dir = get_option('lapack-dir')

_libs = []

_libs += compiler.find_library('lapack', required:true, dirs:[_dir])
_libs += compiler.find_library('blas', required:true, dirs:[_dir])
    
deps += _libs
