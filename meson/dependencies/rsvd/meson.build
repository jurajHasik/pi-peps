# rsvd requires mkl
if not get_option('mkl')
    error('RSVD requires MKL. Please reconfigure by passing -Dmkl=true  and eventually -Dmkl-dir=/path/to/mkl.')
endif

include_dirs += _rsvd_include # defined in the meson.build in the bundled subdir
source_files += _rsvd_source_files # defined in the meson.build in the bundled subdir
