# a custom arpack is required

_dir = get_option('arpack-dir')

if _dir == '' # rely on pkg-config
    arpack = dependency('arpack', required: true)
else
    arpack = declare_dependency(include_directories:
                                include_directories(_dir+'/include/arpack'),
                                link_args:['-L'+_dir+'/lib64', '-larpack'])
endif
# TODO: verify that the version of arpack is correct and has been compiled with the proper flags


deps += arpack

