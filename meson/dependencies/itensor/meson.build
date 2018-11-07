# TODO: use build options like -D itensor_dir
_dir = get_option('itensor-dir')
itensor = declare_dependency(
    include_directories:
      include_directories(_dir),
    link_args:['-L' + _dir + '/lib', '-litensor'])

deps += itensor
