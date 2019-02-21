_opt = 'itensor'

_dir = get_option(_opt+'-dir')

_includedir = include_directories(_dir)

# check that we can actually compile and link
    
compiler = meson.get_compiler('cpp')

_code = '''
#include <itensor/config.h>
int main(){}
'''

_compiles = compiler.compiles(_code, name:
                              _opt.to_upper()+' compile flags check',
                              include_directories:_includedir)

if not _compiles
    error('\n\nCannot find suitable installation for '+_opt.to_upper())
endif


_libpath = _dir+'/lib'

_lib = compiler.find_library('itensor', dirs:_libpath, required:true)


itensor = declare_dependency(
    include_directories: _includedir,
    dependencies:_lib)

deps += itensor
