_source = meson.source_root()+'/examples/opt-fu-adaptive/opt-fu-adaptive.cc'

_exe_name = _source.split('/')[-1].split('.cc')[0]

_dir_name = meson.current_source_dir().split('/')[-1]

_rpath = get_option('prefix')+'/'+get_option('libdir')

exe = executable(_exe_name,
           _source,
           dependencies: our_lib_dep,
           build_by_default: get_option('enable-tests'),
           )

# copy the files in the build dir
_files = ['simulation-fu-HB_2X2_AB.json',
          'reference_output_HB_2X2_AB.in.energy.dat']

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
test('opt-fu-hb-1x2', 
	compare_ll, 
	args:[
		exe.full_path(), 
        	_prefix+'/'+'output_HB_2X2_AB.in.energy.dat',
        	_prefix+'/'+'reference_output_HB_2X2_AB.in.energy.dat',
		'1e-4',
		_prefix+'/'+'simulation-fu-HB_2X2_AB.json'
	],
	workdir: meson.current_build_dir(),
	suite: ['integration'],
	is_parallel : false
)

