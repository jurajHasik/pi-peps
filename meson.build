project('pi-peps','cpp','c',
        default_options : ['cpp_std=c++14',
			   'buildtype=debug',
			   'warning_level=3'],
        version: '1.0')

# useful variables
project_name = meson.project_name()
project_version = meson.project_version()

short_name = 'pi-peps'

include_dirs = []
source_files = []
deps         = []

# everything related to header files:
# - where to look for them (populating variable include_dirs)
# - list which headers should be installed
subdir('include')

# populate the variable source_files
subdir('src')

# deal with bundled dependencies
subdir('bundled')

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

subdir('tests')

subdir('pkgconfig')
