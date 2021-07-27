#!/bin/bash

# default variables
rung16=g16
input=$1

echo -e $"-  RUNNING $input WITH G16 \n"

# run gaussian if com file supplied
if [ -z "$input" ]
then
   echo "NO INPUT!"
else
   $rung16 $input
fi
