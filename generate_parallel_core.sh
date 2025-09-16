#!/bin/sh

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 v e o core cores sito"
    exit 1
fi

v="$1"
emin="$2"
emax="$3"
o="$4"
core="$5"
cores="$6"
sito="$7"


      


for i in $(seq 0 $((1024 / cores - 1))); do
    idx=$((core * 1024 / cores + i))
  
    start_time=$(date +%s%3N)

    geng "$v" "$emin":"$emax" "$idx"/1024 -cq | tee output/"$o"/"$core".g6  | showg -a | ./sita/$sito | ./save.sh output/"$o"/"$core".g6 >> output/"$o"/"$core".out
    # geng "$v" 0:"$e"  "$idx"/1024 -cq | tee output/"$o"/"$core".g6  | showg -a | ./sita/$sito |./save.sh output/"$o"/"$core".g6 >> output/"$o"/"$core".out
  
  
    end_time=$(date +%s%3N)
    elapsed_ms=$((end_time - start_time))
    mins=$((elapsed_ms / 60000))
    secs=$(( (elapsed_ms % 60000) / 1000 ))
    ms=$((elapsed_ms % 1000))
    echo "Core: $core     ran geng $v $emin:$emax $idx/1024 in ${mins}m ${secs}s ${ms}ms"
done

echo "Core $core finished."