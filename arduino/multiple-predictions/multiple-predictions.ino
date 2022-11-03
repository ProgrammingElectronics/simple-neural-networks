#include <lamlia.h>

float weights[] = {0.3, 0.2, 0.9};
float wlrec[] = {0.65, 0.8, 0.9};
float input = wlrec[0];
float output[NUM_ELEMENTS];


void neural_network(float input, float weights[], float buffer[], int len)
{
    scalar_multiply_array(input, weights, buffer, len);
}


void setup() {
  Serial.begin(9600);
  neural_network(input, weights, output, NUM_ELEMENTS);
  print_array_horizontal(output, NUM_ELEMENTS);
}

void loop(){

}