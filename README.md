# diffur_parallel

_compile:_  
mpicxx grid.cpp main.cpp -fopenmp -std=c++98 -O3 -o main -Wall  
_run:_  
mpiexec -n <NPROC> ./main
