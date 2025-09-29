#!/usr/bin/env bash

#clean cython build
rm -rf ./build
rm -rf ./dist
rm -rf ./harpia.egg-info
rm -rf ./__pycache__
find harpia -iname *.so -exec rm {} \;
find harpia -iname *.c -exec rm {} \;
find harpia -iname *.cpp -exec rm {} \;

#clean Makefile build
#rm -rf obj Test


