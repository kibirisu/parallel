#!/bin/sh
# Uruchamianie równoległe geng + showg + integral checker w partiach

N=8                # liczba partii (i równoległych procesów)
NODES=10           # liczba wierzchołków
EDGES=8           # liczba krawędzi
# PROG=./sequential.out
# PROG=./parallel_gpu.out
# PROG=./parallel_cpu.out
PROG=./new_gpu.out

# katalog na wyniki
# OUTDIR=results_parts
# mkdir -p "$OUTDIR"

echo "Start: dzielenie na $N partii (n=$NODES, e=$EDGES)"

i=0
while [ "$i" -lt "$N" ]; do
    echo "  -> uruchamiam partię $i/$N"
    # geng -c "$NODES" "$EDGES" "$i/$N" \
    geng -c "$NODES" "$i/$N" \
        | showg -a \
        | "$PROG" &
        # | showg -a \
        # | "$PROG" > "$OUTDIR/part_$i.out" &
    i=$((i+1))
done

# czekaj na wszystkie równoległe joby
wait

echo "Wszystkie partie zakończone."

# scal wyniki (opcjonalne)
# cat "$OUTDIR"/part_*.out > "$OUTDIR"/all_results.out
# echo "Scalone wyniki zapisane w $OUTDIR/all_results.out"
