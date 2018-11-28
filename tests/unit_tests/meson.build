gtest = dependency('gtest', main: true,
                   fallback : ['gtest', 'gtest_main_dep'],
                   required: false)
gtest_without_main = dependency('gtest', main: false,
                                fallback : ['gtest', 'gtest_dep'],
                                required: false)

test('example',
     executable('example','example.cc',
                dependencies:[gtest,our_lib_dep])
)

test('lin-system',
     executable('lin-sys','test-lin-sys.cc',
                dependencies:[gtest,our_lib_dep])
)

test('svd-solver-gesdd',
     executable('gesdd-svd-solver','test-gesdd-svd-solver.cc',
                dependencies:[gtest,our_lib_dep])
)