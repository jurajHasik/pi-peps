_dir = get_option('mkl-dir')

if _dir == '' # rely on pkg-config
    mkl = dependency('mkl-dynamic-lp64-iomp', required: true)
else
    libdir=_dir+'/lib/intel64_lin'
    omplibdir=_dir + '/../compiler/lib/intel64_lin'
    includedir= _dir+'/include'

    mkl = declare_dependency(include_directories:
                                include_directories(includedir),
                                link_args:['-L'+libdir,
                                           '-lmkl_intel_lp64',
                                           '-lmkl_intel_thread',
                                           '-lmkl_core',
                                           '-L'+omplibdir,
                                           '-liomp5',
                                           '-lpthread',
                                           '-lm',
                                           '-ldl'])

endif

deps += mkl
