# itensor is required
subdir('itensor')

# must have the same name of the option in meson_options.txt
# and the folder must have the same name
# 
optional_deps = ['mkl','arpack']

foreach _opt : optional_deps
    _wanted = get_option(_opt)
    _dir = get_option(_opt+'-dir')
    # the willingness to use a library can be specified
    # from command line either with the option -Dopt
    # or -Dopt-dir
    # there is no need to use both
    if not _wanted
        message('Compiling without '+ _opt.to_upper())
    else
        # first, we source the subdir where we test we can
        # compile and link and we update the list deps
        subdir(_opt)
        # everything went fine and we can define a variable
        # that will be used in the generation of the
        # config file where we log the features of
        # our project
        set_variable('with_'+_opt, true)
    endif
endforeach

# bundled dependendencies are dealt with in a different way
# since the directory, for example, is well known
_bundled = ['rsvd','lbfgs']

foreach _opt : _bundled
    _wanted = get_option(_opt)
    if not _wanted
        message('Compiling without ' + _opt.to_upper())
    else
        message('Using bundled ' + _opt.to_upper())
        # inside the folder we add the source files, include dirs
        # and we need to set the variable 'with_name' to true or false
        # depending also on the other required libraries
        subdir(_opt)
    endif
endforeach

optional_deps += _bundled
