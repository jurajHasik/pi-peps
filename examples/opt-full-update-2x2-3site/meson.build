_source = 'opt-full-update-2x2-3site.cc'

_exe_name = _source.split('.cc')[0]

_dir_name = meson.current_source_dir().split('/')[-1]

_rpath = get_option('prefix')+'/'+get_option('libdir')

executable(_exe_name,
           _source,
           dependencies: our_lib_dep,
           build_by_default: get_option('build-examples'),
           install:true,
           install_dir:'examples/'+_dir_name,
           install_rpath:_rpath)


# copy the files in the build dir
_files = [
'simulation-2site-cholesky.json',
'simulation-2site-pseudoinv.json',
'simulation-AKLT_2x2_ABCD.json',
'simulation-AKLT_2x2_AB.json',
'simulation-HB_2x2_ABCD.json',
'simulation-HB_2x2_AB.json',
'simulation-ISING_2x2_ABCD.json',
'simulation-ISING_2x2_AB.json',
'simulation-J1J2_2x2_ABCD.json',
'simulation-LADDERS_2x2_ABCD.json']

foreach _f : _files
    if(meson.version() >= '0.47')
        configure_file(output:_f,
                       input:_f,
                       copy:true,
                       install:true,
                       install_dir:'examples/'+_dir_name)
    else
        configure_file(output:_f,
                       input:_f,
                       configuration:configuration_data(),
                       install:true,
                       install_dir:'examples/'+_dir_name)
    endif
endforeach
        
        
