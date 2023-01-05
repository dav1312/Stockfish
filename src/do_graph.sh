printf "position startpos\ngo nodes 100\nucinewgame" | ./stockfish
sort -u -o t t && cat header t footer > graph.dot && dot -Tsvg -o graph.svg graph.dot