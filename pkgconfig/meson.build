### generate a .pc file 
pkg = import('pkgconfig')
pkg.generate(libraries: our_lib,
             libraries_private: deps,
             subdirs : 'include',
             version : project_version,
             name : project_name+' library',
             filebase : project_name,
             description : 'C++ library built on top of ITensor for running iPEPS simulations of spin systems for arbitrary unit cells')

