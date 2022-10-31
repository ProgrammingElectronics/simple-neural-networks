/*
From https://www.andreinc.net/2021/01/20/writing-your-own-linear-algebra-matrix-library-in-c
*/

#ifndef lamlia_h
#define lamlia_h

typedef struct ll_mat_s {
  unsigned int num_rows;
  unsigned int num_cols;
  double ** data;
  int is_square;
} ll_mat;

#endif