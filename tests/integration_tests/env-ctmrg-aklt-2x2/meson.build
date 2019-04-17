_source = meson.source_root()+'/examples/env-ctmrg/env-ctmrg.cc'

_exe_name = _source.split('/')[-1].split('.cc')[0]

_dir_name = meson.current_source_dir().split('/')[-1]

_rpath = get_option('prefix')+'/'+get_option('libdir')

exe = executable(_exe_name,
           _source,
           dependencies: our_lib_dep,
           build_by_default: get_option('enable-tests'),
           )

# copy the files in the build dir
_files = ['AKLT_2x2_ABCD.in',
          'simulation-svd-gesdd.json',
          'reference_output.in.energy.dat']

foreach _f : _files
    if(meson.version() >= '0.47')
        configure_file(output:_f,
                       input:_f,
                       copy:true
                       )
    else
        configure_file(output:_f,
                       input:_f
                       )
    endif
endforeach

#test('env-ctmrg-aklt-2x2', exe, args:['simulation-svd-gesdd.json'], workdir: meson.current_build_dir())

_prefix = meson.current_build_dir()
test('env-ctmrg-aklt-2x2', 
	compare, 
	args:[
		exe.full_path(), 
        	_prefix+'/'+'output.in.energy.dat',
        	_prefix+'/'+'reference_output.in.energy.dat',
		_prefix+'/'+'simulation-svd-gesdd.json'
	],
	workdir: meson.current_build_dir(),
	suite: ['integration']
)

