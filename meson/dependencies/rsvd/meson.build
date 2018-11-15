# rsvd requires mkl
if not get_variable('with_mkl')
    error('RSVD requires MKL. Please reconfigure by either passing -Dmkl=true or -Dmkl-dir=/path/to/mkl.')
endif

set_variable('with_rsvd', true)

include_dirs += _rsvd_include # defined in the meson.build in the bundled subdir
source_files += _rsvd_source_files # defined in the meson.build in the bundled subdir
