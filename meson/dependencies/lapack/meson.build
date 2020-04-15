# lapack
_opt= 'lapack'
_dir= get_option(_opt+'-dir')
compiler = meson.get_compiler('cpp')

_libs=[]
if _dir == '' # rely on pkg-config
    blas= dependency('blas', required: false)
    # assume no .pc file
    if not blas.found()
       blas = compiler.find_library('blas')
    endif
    _libs += blas 
    lapack= dependency('lapack', required: false)
    if not lapack.found()
       lapack = compiler.find_library('lapack')
    endif
    _libs += lapack
else
    # assume conda
    _condalib= _dir+'/lib'
    _includedir= include_directories(_dir+'/include')

    _tmp_libs= []
    _lapack_libs= ['blas','lapack'] 
    foreach _lib : _lapack_libs
        _tmp_libs += compiler.find_library(_lib,
                                   dirs:[_condalib], required:true )
    endforeach

    _libs= declare_dependency(include_directories:
                              _includedir,
                              compile_args:'-I'+_dir+'/include',
                              dependencies:_tmp_libs)
endif
#_libs += compiler.find_library('cblas', required:true)

deps += _libs
