#include <lamlia.h>


float weights[NUM_ELEMENTS][NUM_ELEMENTS] = {{0.1, 0.1, -0.3},
                     {0.1, 0.2, 0.0},
                     {0.0, 1.3, 0.1}
};

//  data
float toes[4] = {8.5, 9.5, 9.9, 9.0};
float wlrec[4] = {0.65, 0.8, 0.8, 0.9};  // Win-loss record
float nfans[4] = {1.2, 1.3, 0.5, 1.0};

float input[NUM_ELEMENTS] = {toes[0], wlrec[0], nfans[0]};

float output[NUM_ELEMENTS];

void neural_network(float input[], float weights[][3], float buffer[], int len)
{
  for (int i = 0; i < len; i++){
    float tempBuffer[len];
    //Elementwise multiplication
    elementwise_multiplication(input, weights[i], tempBuffer, len);
    //Sum vector
    //Save sum in output
    buffer[i] = sum_array(tempBuffer, len);      
  }
}


void setup()
{
  Serial.begin(9600);

  neural_network(input, weights, output, NUM_ELEMENTS);
  print_array_horizontal(output, NUM_ELEMENTS);

}

void loop()
{
}