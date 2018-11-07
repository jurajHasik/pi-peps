project('peps-cluster-ctm','cpp',
        default_options : ['cpp_std=c++14',
			   'buildtype=debug',
			   'warning_level=3',
                           'cpp_args=-w'],
        version: '1.0')

# useful variables
project_name = meson.project_name()
project_version = meson.project_version()

include_dirs = []
source_files = []
deps         = []

# everything related to header files:
# - where to look for them (populating variable include_dirs)
# - list which headers should be installed
subdir('include')

# populate the variable source_files
subdir('src')

# dependencies and configurations
# populate the variable deps with all the dependencies
subdir('meson')

our_lib = shared_library(project_name,
                         source_files,
                         dependencies: deps,
                         include_directories: include_dirs,
                         version: project_version,
                         install: true)

our_lib_dep = declare_dependency(dependencies: deps,
                                 link_with: our_lib,
                                 include_directories: include_dirs)

subdir('examples')


### generate a .pc file 
pkg = import('pkgconfig')
pkg.generate(libraries: our_lib,
             libraries_private: deps,
             subdirs : 'include',
             version : project_version,
             name : project_name+' library',
             filebase : project_name,
             description : 'A library that...')
