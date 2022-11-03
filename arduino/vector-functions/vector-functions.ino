#include <AUnit.h>

#define NUM_ELEMENTS 3

float test_arr_1[NUM_ELEMENTS] = {0.1, 0.20, 0};
float test_arr_2[NUM_ELEMENTS] = {8.5, 0.65, 1.2};

/******************************************
 * Array Functions
 ******************************************/
void elementwise_multiplication(float arr_1[], float arr_2[], float buffer[], int len = NUM_ELEMENTS)
{
  for (int i = 0; i < len; i++)
  {
    buffer[i] = arr_1[i] * arr_2[i];
  }
}

void elementwise_addition(float arr_1[], float arr_2[], float buffer[], int len = NUM_ELEMENTS)
{
  for (int i = 0; i < len; i++)
  {
    buffer[i] = arr_1[i] + arr_2[i];
  }
}

float sum_array(float arr[], int len = NUM_ELEMENTS)
{
  float output = 0;

  for (int i = 0; i < len; i++)
  {
    output += arr[i];
  }

  return output;
}

float average_array(float arr[], int len = NUM_ELEMENTS)
{
  float sum = sum_array(arr);
  return sum / NUM_ELEMENTS;
}

float dot_product(float arr_1[], float arr_2[], float buffer[], int len = NUM_ELEMENTS)
{
  elementwise_multiplication(arr_1,arr_2,buffer);
  return sum_array(buffer);
}


/******************************************
 * TESTS
 ******************************************/
test(elementwise_multiplication_stores_correct_answer_in_buffer)
{
  float expected[NUM_ELEMENTS] = {0.85, 0.13, 0.00};
  float under_test[NUM_ELEMENTS];

  elementwise_multiplication(test_arr_1, test_arr_2, under_test, NUM_ELEMENTS);

  for (int i = 0; i < NUM_ELEMENTS; i++)
  {
    assertEqual(expected[i], under_test[i]);
  }
}

test(elementwise_addition_stores_correct_answer_in_buffer)
{
  float expected[NUM_ELEMENTS] = {8.6, 0.850, 1.2};
  float under_test[NUM_ELEMENTS];

  elementwise_addition(test_arr_1, test_arr_2, under_test, NUM_ELEMENTS);

  for (int i = 0; i < NUM_ELEMENTS; i++)
  {
    assertNear(expected[i], under_test[i], 0.02);
  }
}

test(sum_array)
{
  float expected = 0.3;
  float under_test = sum_array(test_arr_1);
  assertEqual(under_test, expected);
}

test(average_array)
{
  float expected = 0.1;
  float under_test = average_array(test_arr_1);
  assertEqual(under_test, expected);
}

test(dot_product)
{
  float expected = 0.98;
  float test_buffer[NUM_ELEMENTS];
  float under_test = dot_product(test_arr_1, test_arr_2, test_buffer);
  assertEqual(under_test, expected);
}

void setup()
{
  Serial.begin(9600);
}

void loop()
{
  aunit::TestRunner::run();
}