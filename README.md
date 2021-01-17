# hpc_benchmarking_c_s1825

In this repository, we compared different implementations of the norm  $d(U) = \sum\sqrt{|u|}$ for a vector U of length `n` and random components between 0 and 1.

We implemented `multi-threading` and `vectorized` computations with the `AVX` instructions set in C.

# How to compile

run the command `gcc -o project project.c -mavx2 -lpthread -lm -Wall`
