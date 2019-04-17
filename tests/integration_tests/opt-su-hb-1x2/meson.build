_source = meson.source_root()+'/examples/opt-su-adaptive/opt-su-adaptive.cc'

_exe_name = _source.split('/')[-1].split('.cc')[0]

_dir_name = meson.current_source_dir().split('/')[-1]

_rpath = get_option('prefix')+'/'+get_option('libdir')

exe = executable(_exe_name,
           _source,
           dependencies: our_lib_dep,
           build_by_default: get_option('enable-tests'),
           )

# copy the files in the build dir
_files = ['simulation-su-HB_2x2_AB.json',
          'reference_output_HB-2x2-AB.in.energy.dat']

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

_prefix = meson.current_build_dir()
test('opt-su-hb-1x2', 
	compare, 
	args:[
		exe.full_path(), 
        	_prefix+'/'+'output_HB-2x2-AB.in.energy.dat',
        	_prefix+'/'+'reference_output_HB-2x2-AB.in.energy.dat',
		_prefix+'/'+'simulation-su-HB_2x2_AB.json'
	],
	workdir: meson.current_build_dir(),
	suite: ['integration'],
	is_parallel : false
)

