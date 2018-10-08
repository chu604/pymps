# MPS program implemented in python by Guochu
- Variational MPS for ground state searching, based on MPO
- time evolving MPS (t-MPS) 

## Time evolution algorithm implemented
- Second order Trotter expansion, for nearest neighbour Hamiltonian
- Fourth order Trotter expansion, for nearest neighbour Hamiltonian
- Second order Trotter expansion, for generic Hamiltonian with only two body terms (not necessarily nearest neighbour, this algorithm is in general slower than previous two algorithms)
- MPO based Runge-Kutta second and fourth algorithm, this is appliable for any Hamiltonian, but will in general be the slowest.

## Installation

1. Download the source file from https://github.com/chu604/pymps
