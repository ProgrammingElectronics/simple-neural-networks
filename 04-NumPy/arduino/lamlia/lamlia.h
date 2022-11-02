/*
From https://www.andreinc.net/2021/01/20/writing-your-own-linear-algebra-matrix-library-in-c
*/

#ifndef lamlia_h
#define lamlia_h

typedef struct ll_mat_s {
  unsigned int num_rows;
  unsigned int num_cols;
  double ** data;
  bool is_square;
} ll_mat;

// Allocate memory for a new matrix
// All elements are 0.0
ll_mat * ll_mat_new(unsigned int num_rows, unsigned int num_cols);

// De-allocate memort for the matrix
void ll_mat_free(ll_mat * matrix);

ll_mat * ll_mat_new(unsigned int num_rows, unsigned int num_cols)
{
  // Guard against 0 dimensional arrays
  if(num_rows == 0)
  {
    LL_ERROR(INVALID_ROWS);
    return NULL;
  }

  if(num_cols == 0)
  {
    LL_ERROR(INVALID_COLUMNS);
    return NULL;
  }

  ll_mat *m = (ll_mat *)calloc(1, sizeof(*m)); //Must cast void pointer ->Not sure this is the Right way

  NULL_POINTER_CHECK(m);
  
  m->num_rows = num_rows;
  m->num_cols = num_cols;
  m->is_square = (num_rows == num_cols) ? true : false;
  m->data = (double **)calloc(m->num_rows, sizeof(*m->data));

  NULL_POINTER_CHECK((ll_mat*)m->data);

  for (int i = 0; i < m-> num_rows; i++)
  {
    ///LEFT OFF HERE...
  }

}


void ll_mat_free(ll_mat * matrix)
{

}


#endif