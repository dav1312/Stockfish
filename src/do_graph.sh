printf "position startpos\ngo nodes 1000\nucinewgame" | ./stockfish
sort -u -o t t && cat header t footer > graph.dot && dot -Tsvg -o graph.svg graph.dot