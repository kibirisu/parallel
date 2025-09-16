#!/bin/sh

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 v e o core cores sito"
    exit 1
fi

v="$1"
e="$2"
o="$3"
core="$4"
cores="$5"
sito="$6"



# rm -f "$o".g6 "$o".adj
# geng "$v" "$e":"$e" -c  | tee "$o".g6 | showg -a  |  awk '!/^Graph/ && NF {printf "%s", $0} /^Graph/ && NR>1 {print ""} END{print ""}' > "$o".adj


# suma=0

# for i in $(seq 0 1024); do
#     count=$(geng "$v" "$e":"$e" "$i"/1024 -cu 2>&1 | grep 'graphs generated' | awk '{print $2}')
#     suma=$((suma + count))
# done
# echo "Total: $suma"


for i in $(seq 0 $((1024 / cores))); do
    idx=$((core * 1024 / $cores + i))

    echo "Starting process $idx"
    geng "$v" "$e":"$e"  "$i"/1024 -cq | tee output/"$o"/"$core".g6  | showg -a | ./sita/sito1.sh >> output/"$o"/"$core".out 
done

echo "Core $core finished."