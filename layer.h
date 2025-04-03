#ifndef LAYER.H
#define LAYER.H
#include "neuron.h"
#include <iostream>

using namespace std;



class Layer {
public:
   Layer(int numRows, int numCols, vector<Neuron> pixels, vector<vector<double>> filter, double bias);
   void printLayerValues() const;
   vector<Neuron> getRow(Layer& layerInput, int rowNum);
   int getSize() const;
   int getFilterSize() const;
   double getBias() const;
   vector<vector<double>> getFilter() const;

  

   vector<vector<Neuron>> getLayer() const;

   


   Layer Convolution(Layer& layerInput, vector<vector<double>>& filter);
   Layer Pooling(Layer& layerInput, int sampleSize);

   double ReLu(double input); 
   int SoftMax(vector<Neuron>);

   Layer FullyConnected(vector<Neuron> flattened, int size);
   int FinalConnected(vector<Neuron> flattened, int size);
   

   vector<Neuron> flatten(Layer& layerInput);




private:
    int Rows; //square layer, so rows = columns
    vector<vector<Neuron>> theLayer;
    vector<vector<double>> filter;
    double bias; 
};

#endif
