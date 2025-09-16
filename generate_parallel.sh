#!/bin/bash

# export OMP_NUM_THREADS=8

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 vertices edges output_dir sito"
    exit 1
fi

v="$1"
emin="$2"
emax="$3"
o="$4"
s="$5"

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


cores=16





monitor() {
    tput clear
    while true; do
        suma=0

        for i in $(seq 0 $((cores-1))); do
            progress=$(head -n 1 output/$o/progress_$i 2>/dev/null || echo 0)
            bar=$(printf "%-${progress}s" "#" | tr ' ' '#')
            elapsed_ms=$(tail -n 1 output/$o/progress_$i 2>/dev/null || echo 0)

            mins=$((elapsed_ms / 60000))
            secs=$(( (elapsed_ms % 60000) / 1000 ))
            ms=$((elapsed_ms % 1000))

            suma=$((suma + elapsed_ms))
            tput cup $i 0
            printf "Core %d: [%-100s] %3d%%      %dm %ds %dms" "$i" "$bar" "$progress" "$mins" "$secs" "$ms"
        done

        mins=$((suma / 60000))
        secs=$(( (suma % 60000) / 1000 ))
        ms=$((suma % 1000))

        tput cup $(($cores + 1)) 120
        printf "Total: %dm %ds %dms"  "$mins" "$secs" "$ms"

        sleep 0.01
    done
}


monitor 2>/dev/null &
monitor_pid=$!
disown $monitor_pid

pids=()

trap 'echo "Killing all child processes..."; pkill -P $$; pkill geng; kill -9 $monitor_pid; exit 1' INT



for i in $(seq 0 $(($cores-1))); do
    # echo "Starting core $i"

    ./generate_parallel_core.sh "$v" "$emin" "$emax" "$o" "$i" "$cores" "$s" & 
    pids+=($!) 
done


wait "${pids[@]}"

sleep 0.2

kill -9 $monitor_pid 2>/dev/null


for i in $(seq 0 $(($cores-1))); do

    cat output/"$o"/"$i".out >> output/"$o"/final.out
done


find output/$o -type f ! -name 'final.out' -delete


echo -e  "\n\n\n\n\n"