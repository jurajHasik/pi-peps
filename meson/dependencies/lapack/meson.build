# lapack

compiler = meson.get_compiler('cpp')

_libs = []

_libs += compiler.find_library('lapack', required:true)
_libs += compiler.find_library('blas', required:true)
_libs += compiler.find_library('cblas', required:true)

deps += _libs
