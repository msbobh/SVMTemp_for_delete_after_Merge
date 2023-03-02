// See https://aka.ms/new-console-template for more information
using System;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using SVM_Using_SequentialMinimumOptimization;
using SVMPrototype_accord;
using System.ComponentModel.DataAnnotations;
using System.Net.Http.Headers;




Console.WriteLine("Starting testing of SVM polynomial kernel");
Console.WriteLine();

double [][] jaggedArray2 = 
{
    new double[] { 1, 3, 5, 7, 9 },
    new double[] { 0, 2, 4, 6, 9 },
    new double[] { 11, 22, 12, 14,32 }
};

var now = new FeatureNormalization(jaggedArray2);
double[][] foo  = now.result;

// SVM testing section
double[][] X = {
  new double[] { 4,5,7 }, new double[] { 7,4,2 },
  new double[] { 0,6,12 }, new double[] { 1,4,8 },
  new double[] { 9,7,5 }, new double[] { 14,7,0 },
  new double[] { 6,9,12 }, new double[] { 8,9,10 }  };
int[] y = { -1, -1, -1, -1, 1, 1, 1, 1 };

// Display the data
for (int i = 0; i < X.Length; ++i)
{
    Console.Write(y[i].ToString().PadLeft(4) + " | ");
    for (int j = 0; j < X[i].Length; ++j)
    {
        Console.Write(X[i][j].ToString("F1").PadLeft(6));
    }
    Console.WriteLine("");
}

var svm = new SVM_Poly(X, y);
var internalsvmPolyobject = svm.Model;
Console.WriteLine("Evaluating SVM model");
bool[] preds = internalsvmPolyobject.Decide(X);
double[] score = internalsvmPolyobject.Score(X);

int numCorrect = 0; int numWrong = 0;
for (int i = 0; i < preds.Length; ++i)
{
    Console.Write("Predicted (double) : " + score[i] + " ");
    Console.Write("Predicted (int): " + preds[i] + " ");
    Console.WriteLine("Actual: " + y[i]);
    if (preds[i] == true && y[i] == 1) ++numCorrect;
    else if (preds[i] == false && y[i] == -1) ++numCorrect;
    else ++numWrong;
}
double acc = (numCorrect * 100.0) / (numCorrect + numWrong);
Console.WriteLine("Model accuracy = " + acc);

Console.WriteLine("Gaussian SVM");
SVM_Gaussian svm_Gaussian = new SVM_Gaussian(X, y);
var internalSVMGaussianObject = svm_Gaussian.Model;
bool[] predictionsGaussian = internalSVMGaussianObject.Decide(X);
double[] ScoreGaussian = internalSVMGaussianObject.Score(X);
int[] integerReturns = svm_Gaussian.Predict_aka_Decide(X);

numCorrect = 0; numWrong = 0;
for (int i = 0; i < preds.Length; ++i)
{
    Console.Write("Predicted Gaussian (double) : " + ScoreGaussian[i] + " ");
    Console.Write("Predicted Gaussian (int): " + integerReturns[i] + " ");
    Console.WriteLine("Actual: " + y[i]); 
    if (integerReturns[i] == 1 && y[i] == 1) ++numCorrect;
    else if (integerReturns[i] == 0 && y[i] == -1) ++numCorrect;
    else ++numWrong;
}
acc = (numCorrect * 100.0) / (numCorrect + numWrong);
Console.WriteLine("Gaussian Model accuracy = " + acc);

// ********************* Sigmoid based model **********************************
Console.WriteLine("Sigmoid Model SVM");
SVM_Sigmoid SVM_Sigmoid = new SVM_Sigmoid(X, y);    
var internalSigoidObject = SVM_Sigmoid.Model;
int[] predictionsSigmoid = SVM_Sigmoid.Decide(X);
double[] ScoreSigmoid = internalSVMGaussianObject.Score(X);
 
    
numCorrect = 0; numWrong = 0;
for (int i = 0; i < preds.Length; ++i)
{
    Console.Write("Predicted Sigmoid (double) : " + ScoreSigmoid[i] + " ");
    Console.Write("Predicted Sigmoid (int): " + predictionsSigmoid[i] + " ");
    Console.WriteLine("Actual: " + y[i]);
    if (predictionsSigmoid[i] == 1 && y[i] == 1) ++numCorrect;
    else if (predictionsSigmoid[i] == 0 && y[i] == -1) ++numCorrect;
    else ++numWrong;
}
acc = (numCorrect * 100.0) / (numCorrect + numWrong);
Console.WriteLine("Model accuracy = " + acc);
Console.WriteLine("Sigmoid Model SVM");

