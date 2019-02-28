_bt = get_option('buildtype') 
if _bt == 'debugoptimized' or _bt == 'release'
  add_project_arguments('-fp_speculative=safe',
                       language : ['cpp'])
endif
