_opt = 'arpack'
_dir = get_option(_opt+'-dir')

compiler = meson.get_compiler('cpp')

if _dir == '' # rely on pkg-config
    arpack = dependency('arpack', required: false)
    # old versions do not install .pc file
    if not arpack.found()
       arpack = compiler.find_libary('arpack')
    endif
else
    _includedir = include_directories(_dir+'/include/arpack')

    # check that we can actually link against arpack
    

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

    if not _compiles
        error('\n\nCannot find suitable installation for '+_opt.to_upper())
    endif

     _libpath = _dir+'/lib'

     _lib = compiler.find_library('arpack', dirs:_libpath, required:true)

    arpack = declare_dependency(include_directories:
                                _includedir,
                                compile_args:'-I'+_dir+'/include/arpack',
                                dependencies:_lib)
endif

deps += arpack

