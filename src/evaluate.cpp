/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <cstring>   // For std::memset
#include <iomanip>
#include <sstream>

#include "bitboard.h"
#include "evaluate.h"
#include "thread.h"

namespace Stockfish {
namespace {

  // Evaluation class computes and stores attacks tables and other working data
  class Evaluation {

  public:
    Evaluation() = delete;
    explicit Evaluation(const Position& p) : pos(p) {}
    Evaluation& operator=(const Evaluation&) = delete;
    Value value();

  private:
    const Position& pos;
  };


  // Evaluation::value() is the main function of the class. It computes the various
  // parts of the evaluation and returns the value of the position from the point
  // of view of the side to move.
  Value Evaluation::value() {
    assert(!pos.checkers());

    int vv =  pos.count<PAWN>(WHITE) - pos.count<PAWN>(BLACK)
            + (pos.count<BISHOP>(WHITE) - pos.count<BISHOP>(BLACK)) * 3
            + (pos.count<KNIGHT>(WHITE) - pos.count<KNIGHT>(BLACK)) * 3
            + (pos.count<ROOK>(WHITE) - pos.count<ROOK>(BLACK)) * 5
            + (pos.count<QUEEN>(WHITE) - pos.count<QUEEN>(BLACK)) * 9;

    vv *= PawnValueEg;
    vv += Value(2 * (pos.this_thread()->nodes & 14) - 14);

    return  Value(pos.side_to_move() == WHITE ? vv : -vv);
  }

} // namespace


/// evaluate() is the evaluator for the outer world. It returns a static
/// evaluation of the position from the point of view of the side to move.

Value Eval::evaluate(const Position& pos) {
  return Evaluation(pos).value();
}


/// trace() is like evaluate(), but instead of returning a value, it returns
/// a string (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.

std::string Eval::trace(const Position& pos) {

  if (pos.checkers())
      return "Total evaluation: none (in check)";

  Value v = Evaluation(pos).value();

  v = pos.side_to_move() == WHITE ? v : -v; // Trace scores are from white's point of view

  std::stringstream ss;
  ss << "Final evaluation: " << double(v) / PawnValueEg << " (white side)\n";

  return ss.str();
}

} // namespace Stockfish
