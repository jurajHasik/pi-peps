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

ll_out=`tail -n 1 $_out`
ll_ref=`tail -n 1 $_ref`

numdiff -a $_tol -r $_tol -s ' \t\n=,:;<>[](){}^' $ll_out $ll_ref

# test if diff was successfull
if [[ $? -ne 0 ]]
then
    echo "diff failed"
    exit 1
fi
exit 0
