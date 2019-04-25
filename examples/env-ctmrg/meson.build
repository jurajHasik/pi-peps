_source = 'env-ctmrg.cc'

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
_files = [_source,
          'RVB_2x2_ABCD.in',
          'RVB_2x2_AB.in',
          'simulation-svd-itensor.json',
          'simulation-svd-gesdd.json',
          'simulation-svd-gesdd_AB.json',
          'simulation-svd-arpack_AB.json',
          'simulation-svd-rsvd.json',
          'simulation-svd-rsvd_AB.json']

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
        
        
# generate the meson.build to be used to compile the example
# once installed
_cdata = configuration_data()
_cdata.set('XXXX',_exe_name)

configure_file(output:'meson.build',
               input:'meson.build.in',
               install:true,
               install_dir:'examples/'+_dir_name,
               configuration:_cdata)
