_source = 'env-ctmrg.cc'

_exe_name = _source.split('.cc')[0]

executable(_exe_name,
           _source,
           dependencies: our_lib_dep,
           build_by_default: get_option('build-examples'))

# copy the files in the build dir
_files = [
'RVB_2x2_ABCD.in',
'RVB_2x2_AB.in',
'simulation-svd-itensor.json',
'simulation-svd-gesdd.json',
'simulation-svd-gesdd_AB.json',
'simulation-svd-rsvd.json',
'simulation-svd-rsvd_AB.json'
]

foreach _f : _files
    if(meson.version() >= '0.47')
        configure_file(output:_f,
                       input:_f,
                       copy:true,
                       install:true)
    else
        configure_file(output:_f,
                       input:_f,
                       configuration:configuration_data(),
                       install:true)
    endif
endforeach
        
        
