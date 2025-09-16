#!/bin/sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 filename"
    exit 1
fi

file="$1"


while read -r lineno; do
    sed -n "${lineno}p" "$file"
done