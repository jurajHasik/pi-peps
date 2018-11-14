_opt = 'arpack'
_dir = get_option(_opt+'-dir')


if _dir == '' # rely on pkg-config
    arpack = dependency('arpack', required: true)
else
    _includedir = include_directories(_dir+'/include/arpack')
    _largs = ['-L'+_dir+'/lib', '-L'+_dir+'/lib64', '-larpack']

    # check that we can actually link against arpack
    
    compiler = meson.get_compiler('cpp')

    _code = '''extern "C" void dnaupd_(int* ido,
                        char* bmat,
                        unsigned int* n,
                        char* which,
                        unsigned int* nev,
                        const double* tol,
                        double* resid,
                        int* ncv,
                        double* v,
                        int* ldv,
                        int* iparam,
                        int* ipntr,
                        double* workd,
                        double* workl,
                        int* lworkl,
                        int* info);

extern "C" void dneupd_(int* rvec,
                        char* howmany,
                        int* select,
                        double* d,
                        double* di,
                        double* z,
                        int* ldz,
                        double* sigmar,
                        double* sigmai,
                        double* workev,
                        char* bmat,
                        unsigned int* n,
                        char* which,
                        unsigned int* nev,
                        double* tol,
                        double* resid,
                        int* ncv,
                        double* v,
                        int* ldv,
                        int* iparam,
                        int* ipntr,
                        double* workd,
                        double* workl,
                        int* lworkl,
                        int* info);

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

    arpack = declare_dependency(include_directories:
                                _includedir,
                                link_args: _largs)
endif

deps += arpack

