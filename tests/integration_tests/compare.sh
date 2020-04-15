#!/bin/bash

# takes at lease three arguments:
# 1. the executable to run
# 2. the reference output file
# 3. all the needed arguments

_exe=$1; shift
_out=$1; shift
_ref=$1; shift
_rest=$@

$_exe $_rest

# test if run was successfull
if [[ $? -ne 0 ]]
then
    echo "run failed"
    exit 1
fi

numdiff -a 1e-8 -r 1e-8 -s ' \t\n=,:;<>[](){}^' $_out $_ref

# test if diff was successfull
if [[ $? -ne 0 ]]
then
    echo "diff failed"
    exit 1
fi

exit 0
