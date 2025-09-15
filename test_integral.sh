#!/bin/sh
# Minimalny test integralnych grafów

PROG=./parallel
TMPDIR=tmp_test_graphs
mkdir -p "$TMPDIR"

echo "== Test integralnych grafów =="

# Funkcja pomocnicza: generuj, sprawdź, pokaż wynik
runtest() {
    N=$1
    DESC=$2
    FILE="$TMPDIR/test_${N}.g6"

    echo
    echo ">>> Test: $DESC (n=$N)"
    geng "$N" -c > "$FILE"

    # uruchom filtr, wypisz ile znalazło
    COUNT=$($PROG "$FILE" | wc -l | tr -d ' ')
    echo "Integralnych znaleziono: $COUNT"
    echo "Przykłady wyników:"
    head -n 3 "$FILE" | $PROG -v 3 -
}

# K2 (n=2) - trywialny graf pełny
runtest 2 "K2"

# K3 (n=3) - pełny trójkąt, znany integralny
runtest 3 "K3"

# C4 (cykl na 4) - integralny
runtest 4 "C4"

# C6 (cykl na 6) - integralny
runtest 6 "C6"

echo
echo "== Test zakończony =="
