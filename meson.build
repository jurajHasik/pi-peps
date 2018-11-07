project('peps-cluster-ctm','cpp',
        default_options : ['cpp_std=c++14',
			   'buildtype=debug',
			   'warning_level=3',
                           'cpp_args=-w'],
        version: '0.1.0')

subdir('meson')

common = static_library('common_part',
                        sources: files(['cluster-ev-builder.cc', 
                                        'engine.cc',           
                                        'full-update-TEST.cc', 
                                        'full-update.cc', 
                                        'simple-update_v2.cc', 
                                        'models.cc',
                                        'ctm-cluster-env_v2.cc', 
                                        'ctm-cluster-io.cc',
                                        'ctm-cluster.cc',
                                        'mpo.cc',
                                        'su2.cc']),
                        dependencies: [itensor,arpack,mkl],
                        install:false)


mains = ['opt-full-update-2x2-3site.cc',
         'opt-su-3site.cc']

foreach _source : mains
    _exe_name = _source.split('.cc')[0]
    executable(_exe_name,
               _source,
               dependencies: [itensor,mkl,arpack],
               link_with: common,
               install: true)
endforeach


