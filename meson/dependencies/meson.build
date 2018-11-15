# itensor is required
subdir('itensor')

# must have the same name of the option in meson_options.txt
# and the folder must have the same name
# 
optional_deps = ['arpack','mkl']

foreach _opt : optional_deps

    _wanted = get_option(_opt)
    _dir = get_option(_opt+'-dir')

    # the willingness to use a library can be specified
    # from command line either with the option -Dopt
    # or -Dopt-dir
    # there is no need to use both
    if not _wanted and _dir == ''
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

# here we set the variables for bundled dependendencies
# used for generating the config file
# the source files and headers are managed in the proper
# subdir of bundled
_bundled = ['rsvd']

foreach _opt : _bundled
    _wanted = get_option(_opt)
    message(_opt.to_upper() + 'feature: '+ '@0@'.format(_wanted))
    set_variable('with_'+_opt, _wanted)
endforeach

optional_deps += _bundled
