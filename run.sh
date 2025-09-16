#!/bin/sh

name=$(tr -dc 'a-zA-Z' </dev/urandom | head -c7)

time ./generate_parallel.sh 10 20 20 $name program1