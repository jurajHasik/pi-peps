_cdata = configuration_data()

# optional_deps is defined in meson.build in the dependencies folder
foreach _opt : optional_deps
    _v = 'PEPS_WITH_'+ _opt.to_upper()
    _cdata.set(_v, get_option(_opt))
endforeach


code = '''_Pragma("GCC diagnostic push")
  _Pragma("GCC diagnostic ignored \"-Wextra\"")
  _Pragma("GCC diagnostic ignored \"-Wunknown-pragmas\"")
  _Pragma("GCC diagnostic ignored \"-Wpragmas\"")
  int main() { return 0; }
  _Pragma("GCC diagnostic pop")
'''

_pragma = meson.get_compiler('cpp').compiles(code, name: 'pragma check')

_cdata.set('COMPILER_HAS_DIAGNOSTIC_PRAGMA', _pragma)


configure_file(output: 'config.h',
               input: 'config.h.in',
               configuration: _cdata,
               install: true,
               install_dir:'include/p-ipeps')
