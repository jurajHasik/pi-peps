_bt = get_option('buildtype')

if _bt == 'debugoptimized' or _bt == 'release'
  add_project_arguments(['-funroll-loops',
                        '-fstrict-aliasing',
                        '-O3'],
                       language : ['cpp'])
endif

if _bt == 'release'
  add_project_arguments(['-DNDEBUG'],
                       language : ['cpp'])
endif

_supported_compilers = ['clang', 'gcc', 'intel']

foreach _c : _supported_compilers
    if meson.get_compiler('cpp').get_id() == _c
        subdir(_c)
    endif
endforeach

