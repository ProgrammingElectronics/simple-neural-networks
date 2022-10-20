#include <assert.h>

#define NUM_WEIGHTS 3
#define NUM_INPUTS 3

float weights[NUM_WEIGHTS] = {0.1, 0.2, 0};

// data
const float toes[] = {8.5, 9.5, 9.9, 9.0};
const float wlrec[] = {0.65, 0.8, 0.8, 0.9}; // Win-loss record
const float nfans[] = {1.2, 1.3, 0.5, 1.0};

float w_sum(float input[], float weights[])
{

  float output = 0;

  for (int i = 0; i < NUM_INPUTS; i++)
  {
    output += (input[i] * weights[i]);
  }

  return output;
}

float neural_network(float input[], float weights[])
{

  float pred = w_sum(input, weights);

  return pred;
}

void setup()
{
  Serial.begin(9600);
}

void loop()
{

  static boolean complete = false;

  if (!complete)
  {
    float input[NUM_INPUTS] = {toes[0], wlrec[0], nfans[0]};
    float pred = neural_network(input, weights);

    Serial.println(pred);

    complete = true;
  }
}
