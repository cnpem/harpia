#!/usr/bin/env bash
rm -rf ./build
rm -rf ./dist
find harpia -iname *.so -exec rm {} \;
find harpia -iname *.c -exec rm {} \;
find harpia -iname *.cpp -exec rm {} \;
