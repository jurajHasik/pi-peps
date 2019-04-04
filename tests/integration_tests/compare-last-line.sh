#!/bin/bash

# takes at lease three arguments:
# 1. the executable to run
# 2. the reference output file
# 3. all the needed arguments

_exe=$1; shift
_out=$1; shift
_ref=$1; shift
_tol=$1; shift
_rest=$@

$_exe $_rest

# test if run was successfull
if [[ $? -ne 0 ]]
then
    echo "run failed"
    exit 1
fi

tail -n 1 $_out >> out_ll.dat.$$
tail -n 1 $_ref >> ref_ll.dat.$$

numdiff -a $_tol -r $_tol -s ' \t\n=,:;<>[](){}^' out_ll.dat.$$ ref_ll.dat.$$

# test if diff was successfull
if [[ $? -ne 0 ]]
then
    echo "diff failed"
    rm out_ll.dat.$$
    rm ref_ll.dat.$$
    exit 1
fi
rm out_ll.dat.$$
rm ref_ll.dat.$$
exit 0
