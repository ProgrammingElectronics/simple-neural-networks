#ifndef lamlia_util_h
#define lamlia_util_h

enum ERRRORS
{
  INVALID_ROWS,
  INVALID_COLUMNS,
};

void LL_ERROR(byte ERROR_MSG)
{
  switch (ERROR_MSG)
  {
  case INVALID_ROWS:
    Serial.print("Error Message -> ");
    Serial.println("INVALID ROWS");
    break;

  case INVALID_COLUMNS:
    Serial.print("Error Message -> ");
    Serial.println("INVALID_COLUMNS");
    break;

  default:
    break;
  }
}

bool NULL_POINTER_CHECK(ll_mat *ptr)
{
  if (!(ptr))
  {
    Serial.println("That was a null pointer!");
    while(1){};
  }

  return true;
}

#endif