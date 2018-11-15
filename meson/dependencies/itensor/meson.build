_opt = 'itensor'

_dir = get_option(_opt+'-dir')

_includedir = include_directories(_dir)
_largs = ['-L' + _dir + '/lib', '-litensor']

# check that we can actually compile and link
    
compiler = meson.get_compiler('cpp')

_code = '''
#include <itensor/config.h>
int main(){}
'''

_compiles = compiler.compiles(_code, name:
                              _opt.to_upper()+' compile flags check',
                              include_directories:_includedir)

_links = compiler.links(_code, name:
                        _opt.to_upper()+' link flags check',
                        include_directories:_includedir,
                        args:_largs)

if not (_compiles and _links)
    error('\n\nCannot find suitable installation for '+_opt.to_upper())
endif

itensor = declare_dependency(
    include_directories:
    _includedir,
    link_args:_largs)

deps += itensor
