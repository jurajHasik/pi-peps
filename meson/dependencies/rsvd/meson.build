# a custom arpack is required

_dir = get_option('rsvd-dir')

if _dir == '' # rely on pkg-config
    rsvd = dependency('arpack', required: true)
else
    rsvd = declare_dependency(include_directories:
                                include_directories(_dir),
                                link_args:['-L'+_dir, '-lrsvd'])
endif

deps += rsvd
