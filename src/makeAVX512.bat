@echo off
REM x64 builds begin
SET PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%

Title "x86-64-avx512"
make clean
mingw32-make profile-build ARCH=x86-64-avx512 COMP=mingw CXX=x86_64-w64-mingw32-g++ -j %Number_Of_Processors%
strip stockfish.exe
ren stockfish.exe "Big-SF 200324-avx512.exe"

make clean
REM x64 builds end
Pause

