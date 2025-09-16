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


      
start_time=$(date +%s%3N)


for i in $(seq 0 $((1024 / cores - 1))); do
    idx=$((core + i * co10res))
  

    geng "$v" "$emin":"$emax" "$idx"/1024 -cq | tee output/"$o"/"$core".g6  | showg -a  2>/dev/null | ./sita/$sito | ./save.sh output/"$o"/"$core".g6 >> output/"$o"/"$core".out  2>/dev/null
 
    


    end_time=$(date +%s%3N)
    elapsed_ms=$((end_time - start_time))

    
    percent=$(( (i+1)*100 * cores / 1024 ))

    echo $percent > output/"$o"/progress_$core
    # echo "${mins}m ${secs}s ${ms}ms"  >> output/"$o"/progress_$core
    echo $elapsed_ms >> output/"$o"/progress_$core
    # echo $percent   > output/"$o"/progress_$core

    # echo "Core: $core     ran geng $v $emin:$emax $idx/1024 in ${mins}m ${secs}s ${ms}ms"
done
echo 100 > output/"$o"/progress_$co

# echo "Core $core finished."