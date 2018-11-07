# TODO: use build options like -D itensor_dir
itensor = declare_dependency(
    include_directories:
      include_directories('/scratch/src/itensor-custom'),
    link_args:['-L/scratch/src/itensor-custom/lib', '-litensor'])

deps += itensor
