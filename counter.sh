#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 v e "
    exit 1
fi

v="$1"
e="$2"
# o="$3"

echo "Vertices: $v"
echo "Edges: $e"
# echo "Output: $o"


mkdir -p output

suma=0

for i in $(seq 0 1024); do
    count=$(geng "$v" "$e":"$e" "$i"/1024 -cu 2>&1 | grep 'graphs generated' | awk '{print $2}')
    suma=$((suma + count))
done
echo "Total: $suma"

