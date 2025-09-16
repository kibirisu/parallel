#!/bin/sh

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 vertices edges output_dir sito"
    exit 1
fi

v="$1"
e="$2"
o="$3"
s="$4"

echo "Vertices: $v"
echo "Edges: $e"
echo "Sito: $s"
echo "Output: $o"

mkdir -p output/$o
rm -rf output/$o/*

# rm -f "$o".g6 "$o".adj
# geng "$v" "$e":"$e" -c  | tee "$o".g6 | showg -a  |  awk '!/^Graph/ && NF {printf "%s", $0} /^Graph/ && NR>1 {print ""} END{print ""}' > "$o".adj


# suma=0

# for i in $(seq 0 1024); do
#     count=$(geng "$v" "$e":"$e" "$i"/1024 -cu 2>&1 | grep 'graphs generated' | awk '{print $2}')
#     suma=$((suma + count))
# done
# echo "Total: $suma"

trap 'echo "Killing all child processes..."; pkill -P $$; pkill geng; exit 1' INT

cores=16

for i in $(seq 0 $(($cores-1))); do
    echo "Starting core $i"

    ./generate_parallel_core.sh "$v" "$e" "$o" "$i" "$cores" "$s" &
done

wait

for i in $(seq 0 $(($cores-1))); do

    cat output/"$o"/"$i".out >> output/"$o"/final.out
done


