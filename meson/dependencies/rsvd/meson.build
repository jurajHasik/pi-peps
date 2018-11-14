_opt = 'rsvd'
_dir = get_option(_opt+'-dir')

if _dir == '' # rely on pkg-config
    arpack = dependency('rsvd', required: true)
else
    _includedir = include_directories(_dir)
    _largs = ['-L'+_dir, '-lrsvd']

    # check that we can actually link against rsvd
    
    compiler = meson.get_compiler('cpp')

    _code = '''
    int main(){
    }'''

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

    rsvd = declare_dependency(include_directories: _includedir,
                              link_args: _largs, sources:
                              files(meson.source_root()+'/src/rsvd-solver.cc'))
    
endif

deps += rsvd

