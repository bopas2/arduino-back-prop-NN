#include <math.h>

const int input = 2;
const int hidden = 4;
const int output = 1;
const int biasedIn = input + 1;
const int biasedHid = hidden + 1;

const int numOfTrainingSets = 4;

const float learningRate = .1;
const float lambda = .1;

float theta1[biasedIn][hidden];
float theta2[biasedHid][output];


const float inputData[numOfTrainingSets][input] = {
  {1, 1},
  {1, 0},
  {0, 0},
  {0, 1},
};

const float trainingAns[numOfTrainingSets][output] = {
  {1},
  {1},
  {0},
  {1}
};

void setup() {
  randomSeed(analogRead(2));
  Serial.begin(9600);
  randomize(*theta1, biasedIn, hidden);
  randomize(*theta2, biasedHid, output);
  train(*inputData,*trainingAns,2);
}
void loop() {
  //Random weights to break symmetry

}

void train(float * in, float * target, int iters) {
  int z = 0;
  while (z < iters) {
    for (int j = 0; j < 1; j++) {
      float set[1][input];
      float ans[1][output];
      for (int k = 0; k < input; k++) {
        set[0][k] = *(in + j * input + k);
  //      Serial.print(set[0][k]); Serial.print(" ");
      }
     // Serial.println();
      for (int k = 0; k < output; k++) {
        ans[0][k] = *(target + j * output + k);
//        Serial.println(ans[0][k]); 
      }
      float * theta1grad;
      float * theta2grad;
      calcGradient(*set, *ans, theta1grad, theta2grad);
      //      float * one = addOrSubtract(*theta1, theta1grad, biasedIn, hidden, true);
      //      float * two = addOrSubtract(*theta2, theta2grad, biasedHid, output, true);
      //      for (int a = 0; a < biasedIn; a++) {
      //        for (int b = 0; b < hidden; b++) {
      //          theta1[a][b] = *(one + a * hidden + b);
      //        }
      //      }
      //      for (int a = 0; a < biasedHid; a++) {
      //        for (int b = 0; b < output; b++) {
      //          theta2[a][b] = *(two + a * output + b);
      //        }
      //      }
    }
    z++;
  }
}

//creates random weights to break symmetry - correct
void randomize(float  *a, int r, int c) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      float r = (((float)random(100) / 100.0) * 2 * .12) - .12;
      *(a + (i * c) + j) = r;
    }
  }
}

//multiplies two matricies together - correct
float* multiply(float *a, float *b, int ra, int ca, int rb, int cb) {
  float * ans;
  ans = (float *) malloc(ra * cb * sizeof(float));
  for (int i = 0; i < ra; i++) {
    for (int j = 0; j < cb; j++) {
      float sum = 0;
      for (int k = 0; k < ca; k++) {
        sum += *(a + (i * ca) + k) * *(b + (k * cb) + j);
      }
      *(ans + (i * cb) + j) = sum;
    }
  }
  return ans;
}

//predicts output based on input
float * feedForward(float * given) {
  float * in = addBias(given, 1, input);
  float * inner = multiply(in, *theta1, 1, biasedIn, biasedIn, hidden);
  for (int i = 0; i < hidden; i++) {
    *(inner + i) = sigmoid(*(inner + i));
  }
  float * biasedA2 = addBias(inner, 1, hidden);
  float * outer = (float *) malloc (output * sizeof(float));
  outer = multiply(biasedA2, *theta2, 1, biasedHid, biasedHid, output);
  for (int i = 0; i < output; i++) {
    *(outer + i) = sigmoid(*(outer + i));
  }
  free(in);
  free(inner);
  free(biasedA2);

  return outer;
}

//gets us the a2 value used for backprop
float * feedForwardA2(float * given) {
  float * in = addBias(given, 1, input);
  float * inner = multiply(in, *theta1, 1, biasedIn, biasedIn, hidden);
  for (int i = 0; i < hidden; i++) {
    *(inner + i) = sigmoid(*(inner + i));
  }
  float * ans = malloc(biasedHid * sizeof(float));
  ans = addBias(inner, 1, biasedHid);
  return ans;
}

//Adds the bias unit to the matrix (extra collumn of ones at start of matrix) - correct
float * addBias(float * in, int rows, int collumns) {
  float * answa = (float *) malloc (rows * (collumns + 1) * sizeof(float));
  for (int i = 0; i < rows; i++) {
    *(answa + (i * (collumns+1))) = 1.0;
  }
  for (int i = 0; i < rows; i++) {
    for (int j = 1; j < collumns+1; j++) {
      *(answa + (i * (collumns+1)) + j) = *(in + (i * (collumns)) + (j - 1));
    }
  }
  return answa;
}

//Removes the bias collumn of a matrix (first collumn) - CORRECT
float * removeBias(float * in, int rows, int collumns) {
  float * answa = (float *) malloc (rows * (collumns - 1) * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 1; j < collumns; j++) {
      *(answa + (i * (collumns - 1)) + (j-1)) = *(in + (i * collumns) + (j));
    }
  }
  return answa;
}

//Sigmoidifies a number - CORRECT
float sigmoid(float in) {
  return 1.0 / (1.0 + exp(-1 * in));
}

//Applies the sigmoid gradient to each unit in the matrix - correct
float * sigmoidGradient(float * in, int rows, int collumns) {
  float * ans;
  ans = malloc(rows * collumns * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < collumns; j++) {
      *(ans + i * collumns + j) = sigmoid(*(in + i * collumns + j)) * (1 - sigmoid(*(in + i * collumns + j)));
    }
  }
  return ans;
}

//Completes the transpose matrix calculation - correct
float * transpose(float * in, int rows, int collumns) {
  float * ans;
  ans = malloc(rows * collumns * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < collumns; j++) {
      *(ans + j * rows + i) = *(in + i * collumns + j);
    }
  }
  return ans;
}

//multiplies each single unit by a constant - correct!
float * dotMultiplyConst(float *in, float mult, int rows, int collumns) {
  float * ans;
  ans = malloc(rows * collumns * sizeof(float));
  for (int g = 0; g < (rows * collumns); g++) {
    *(ans + g) = *(in + g) * mult;
  }
  return ans;
}

//multiplies each unit by its respective counterpart in a same sized matrix - correct!
float * dotMultiply(float *in, float *mult, int rows, int collumns) {
  float * ans;
  ans = malloc(rows * collumns * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < collumns; j++) {
      *(ans + i * collumns + j) = *(in + i * collumns + j) * (*(mult + i * collumns + j));
    }
  }
  return ans;
}

//adds or subtracts two same sized matricies from eachother - CORRECT
float* addOrSubtract(float * a, float * b, int rows, int collumns, boolean subtract) {
  float * ans;
  ans = malloc(rows * collumns * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < collumns; j++) {
      if (subtract)
        *(ans + i * collumns + j) = *(a + i * collumns + j) - *(b + i * collumns + j);
      else
        *(ans + i * collumns + j) = *(a + i * collumns + j) + *(b + i * collumns + j);
    }
  }
  return ans;
}

void calcGradient(float * given, float * actual, float * &theat1gradR, float* &theta2gradR) {
  float * guess = feedForward(given);
  float * theta1grad = malloc(biasedIn * hidden * sizeof(float));
  float * theta2grad = malloc(biasedHid * output * sizeof(float));
  float * cost3 = addOrSubtract(guess, actual, 1, output, true);
  float * cost2 = multiply(cost3, transpose(*theta2, biasedHid, output), 1, output, output, biasedHid);

  cost2 = dotMultiply(cost2, sigmoidGradient(feedForwardA2(given), 1, biasedHid), 1, biasedHid);
  cost2 = removeBias(cost2, 1, biasedHid);
  float * temp2 = transpose(addBias(given, 1, input), 1, input + 1);

  
   float * temp = multiply(temp2, cost2, input + 1, 1, 1, hidden);
   for(int i = 0; i < input+1; i++) {
    for(int j = 0; j < hidden; j++) {
      Serial.print(*(temp+i*hidden+j)); Serial.print(" ");
    }
    Serial.println();
   }
 // theta1grad = addOrSubtract(theta1grad, temp, input, hidden, false);
  //  theta2grad = addOrSubtract(theta2grad, multiply(transpose(given, 1, input), cost3, 1, input, 1, output), biasedIn, hidden, false);
  //  theta1grad = addOrSubtract(theta1grad, dotMultiplyConst(*theta1, lambda, biasedIn, hidden), biasedIn, hidden, false);
  //  theta2grad = addOrSubtract(theta2grad, dotMultiplyConst(*theta2, lambda, biasedHid, output), biasedIn, hidden, false);
  //  theat1gradR = theta1grad;
  //  theta2gradR = theta2grad;
}

