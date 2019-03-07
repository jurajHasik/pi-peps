### generate a .pc file 
pkg = import('pkgconfig')
pkg.generate(our_lib,
             libraries: deps,
             version : project_version,
             name : project_name+' library',
             filebase : project_name,
             install_dir: get_option('prefix'),
             description : 'C++ library built on top of ITensor for running iPEPS simulations of spin systems for arbitrary unit cells')

