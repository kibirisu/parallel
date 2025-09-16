#!/bin/sh

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 v e o"
    exit 1
fi

v="$1"
e="$2"
o="$3"

echo "Vertices: $v"
echo "Edges: $e"
echo "Output: $o"


mkdir -p output

rm -f "$o".g6 "$o".adj
geng "$v" "$e":"$e" -c  | tee "$o".g6 | showg -a  |  awk '!/^Graph/ && NF {printf "%s", $0} /^Graph/ && NR>1 {print ""} END{print ""}' > "$o".adj
