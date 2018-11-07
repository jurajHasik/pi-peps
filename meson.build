project('peps-cluster-ctm','cpp',
        default_options : ['cpp_std=c++14',
			   'buildtype=debug',
			   'warning_level=3' ],
        version: '0.1.0')


mkl = dependency('mkl-dynamic-lp64-iomp')

arpack = dependency('arpack')

itensor = declare_dependency(
    include_directories:
      include_directories('/scratch/src/itensor-custom'),
    link_args:['-L/scratch/src/itensor-custom/lib', '-litensor'])


executable('opt-full-update-2x2-3site',
           files(['opt-full-update-2x2-3site.cc',
                  'cluster-ev-builder.cc',
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
           dependencies: [itensor,mkl,arpack])
