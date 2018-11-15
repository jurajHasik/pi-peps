_cdata = configuration_data()

# optional_deps is defined in meson.build in the dependencies folder
foreach _opt : optional_deps
    _v = short_name.to_upper() + '_WITH_'+ _opt.to_upper()
    _cdata.set(_v, get_variable('with_'+_opt, false))
endforeach

configure_file(output: short_name+'_config.h',
               input: 'config.h.in',
               configuration: _cdata,
               install: true)

include_dirs += include_directories('.')
