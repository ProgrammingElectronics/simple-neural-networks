// Adapted from Grokking Deep Learning Chapter 3

const float weight = 0.1;
const byte NUM_INPUTS = 4;
const float number_of_toes[NUM_INPUTS] = {8.5, 9.5, 10, 9}; 

float neural_network(float input, float weight)
{
  float pred = input * weight;

  return pred;
}

void setup()
{
  Serial.begin(9600);
}

void loop()
{

  static bool complete = false;

  if (!complete)
  {
    float pred = neural_network(number_of_toes[0], weight);
    Serial.println(pred);
    complete = true;
  }
}