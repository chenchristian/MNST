#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip> // Include the header for setprecision
#include "neuron.h"
#include "Layer.h"

using namespace std;

Neuron::Neuron()
{
    value = 0.0;
}

Neuron::Neuron(double pixels)
{
    value = pixels;
    weight = 1.0; //you could make it random to see variety in your softmax
}


double Neuron::getValue() const
{
    return value;
}

double Layer::getBias() const
{
    return bias;
}

double Layer::ReLu(double input)
{
    if(input <= 0)
    {
        return 0;
    }
    else
    {
        return input;
    }
}

int Layer::SoftMax(vector<Neuron> final )
{
    double denominator = 0.0;
    for(int i = 0; i < final.size(); i++)
    {
        denominator += exp(final[i].getValue()); // this is e^x where x is the value of the final neuron
    }


    vector<double> softmax;
    for(int i = 0; i < final.size(); i++)
    {
        double numerator = exp(final[i].getValue());
        softmax.push_back(numerator/denominator);
    }

    int guess = 0;
    double maxValue = 0.0;
    for(int i = 0; i < final.size(); i++)
    {
        if(softmax[i] > maxValue)
        {
            maxValue = softmax[i];
            guess =i;

        }
    }

    return guess;
    

}





Layer::Layer(int numRows, int numCols, vector<Neuron> pixels, vector<vector<double>> filterInput, double biasInput) 
{
    filter = filterInput;
    Rows = numRows;
    bias =  biasInput;
    theLayer.resize(numRows, vector<Neuron>(numCols));


    for(int i = 0; i < numRows; i++)
    {
        for(int j = 0; j < numCols; j++)
        {
            theLayer[i][j] = pixels[numCols * i + j];
        }
    }
}

void Layer::printLayerValues() const
{
     for (const auto &row : theLayer) {
        for (const auto &neuron : row) {
            cout << setw(2) << setfill('0') << neuron.getValue() << " ";
        }
        std::cout << std::endl;
    }
}

vector<Neuron> Layer::getRow(Layer& layerInput, int rowNum)
{
    return theLayer[rowNum];
}

int Layer::getSize() const //returns the size of a row
{
    return Rows;
}

int Layer::getFilterSize() const
{
    return filter.size();
}

vector<vector<double>> Layer::getFilter() const
{
    return filter;
}

vector<vector<Neuron>> Layer::getLayer() const
{
    return theLayer;
}

Layer Layer::Convolution(Layer& layerInput, vector<vector<double>>& filter) //5x5 filter
{
    int newSize = layerInput.getSize() - layerInput.getFilterSize() + 1;
    double bias = layerInput.getBias();

    vector<Neuron> finishedConvo(newSize * newSize); //flattens the 2D vector into 1D
    for(int i = 0; i < newSize; i++) //i is row the big window
    {
        for(int j = 0; j < newSize; j++) //j is the column of the big window
        {
            double filteredSum = 0.0; //Initialize filtered sum for this window
            
            for(int k = 0; k < layerInput.getFilterSize(); k++) //k is the row of the filter
            {
                for(int l = 0; l < layerInput.getFilterSize(); l++) //l is the columns of the filter
                {
                    vector<Neuron> ConvoLayer = layerInput.getRow(layerInput, k+i);
                    Neuron zeroZero = ConvoLayer[l+j];                                           
                    filteredSum = filteredSum + zeroZero.getValue() * filter[k][l];
                }
            }
            filteredSum = ReLu(filteredSum) + bias;
            Neuron newNeuron(filteredSum);
            finishedConvo[newSize * i + j] = newNeuron; //wait im pretty sure I can use a pushback function here, come back to this
        }
    }
    double newBias = 1;
    Layer convolvedLayer(newSize, newSize, finishedConvo, filter, newBias); //filter should be a new one... for pooling
    return convolvedLayer;
}

Layer Layer::Pooling(Layer& layerInput, int sampleSize) //2x2 window size, and it doest not blend over previous window
{
    int newSize = layerInput.getSize() / sampleSize; // Will return the size of the new layer after pooling
    vector<Neuron> finishedPooling(newSize * newSize);

    for(int i = 0; i < newSize; i++) //i is row the big window 
    {
        for(int j = 0; j < newSize; j++) //j is the column of the big window
        {
            double Biggest = 0.0; //Initialize filtered sum for this window

            for(int k = 0; k < sampleSize; k++) //k is the row of the filter
            {
                for(int l = 0; l < sampleSize; l++) //l is the columns of the filter
                {
                    vector<Neuron> firstLayer = layerInput.getRow(layerInput, 2*i+l); //2i to account for the pooling window only selecting new values
                    double value = firstLayer[2*j + k].getValue();

                    if (value > Biggest)
                    {
                        Biggest = value;
                    }
                }
            }
            Neuron newNeuron(Biggest);
            finishedPooling[newSize * i + j] = newNeuron;
        }
    }

    double newBias = 1;
    Layer PooleddLayer(newSize, newSize, finishedPooling, filter, newBias); //filter should be a new one... for pooling
    return PooleddLayer;
}


Layer Layer::FullyConnected(vector<Neuron> flattened, int size)
{
    vector<Neuron> nextFlattened;
    for(int i = 0; i< size; i++)
    {
        double sum = 0.0;
        for(int j = 0; j< size; j++)
        {
            sum += flattened[j].getValue() * flattened[j].getWeight();
        }

        double newSum = sum + bias;

        Neuron output(ReLu(newSum));
        nextFlattened.push_back(output);
    }

    return Layer(size, 1, nextFlattened, filter, bias); //dont need filter or bias but I need to keep data type layer
}

int Layer::FinalConnected(vector<Neuron> flattened, int size)
{
    vector<Neuron> nextFlattened;
    for(int i = 0; i< size; i++)
    {
        double sum = 0.0;
        for(int j = 0; j< size; j++)
        {
            sum += flattened[j].getValue() * flattened[j].getWeight();
        }

        double newSum = sum + bias;

        Neuron output(ReLu(newSum));
        nextFlattened.push_back(output);
    } 

    return SoftMax(nextFlattened);
        
}


vector<Neuron> Layer::flatten(Layer& layerInput)
{
    vector<vector<Neuron>> preFlat = layerInput.getLayer();
    vector<Neuron> flattened;
    for(int i = 0; i< preFlat.size(); i++)
    {
        for(int j = 0; j< preFlat[i].size(); j++)
        {
            flattened.push_back(preFlat[i][j]);
        }
    }

    return flattened;

}

vector<double> ReadFile(string filepath)
{
    ifstream file(filepath);

    vector<double> imageData;
    string line;

    while (getline(file, line)) {
        istringstream iss(line);

        double pixelValue;
        while (iss >> pixelValue) {
            imageData.push_back(pixelValue);
        }
    }

    return imageData;
}

void printImageTest(vector<double> imageData)
{
    // Output pixel values in a 28x28 format
    for (int i = 0; i < imageData.size(); ++i) {
        cout << imageData[i] << " ";
        if ((i + 1) % 28 == 0) {
            cout << endl;
        }
    }
}


int main() 
{

    
    vector<double> imageLayer = ReadFile("/Users/christianchen/Desktop/c++/MNIST/zero.txt");
    
    //printImageTest(vector<double> imageData); 
    
    

    vector<double> filterPartOne = {1,0,1,0,1};
    vector<double> filterPartTwo = {0,1,0,1,0};

    vector<vector<double>> filter;
    for(int i = 0; i < 5; i++)
    {
        if(i % 2 == 0)
        {
            filter.push_back(filterPartOne);
        }
        else
        {
            filter.push_back(filterPartTwo);
        }
    }

    const int numElements = 784; // 28*28

    // converting vector<double> from image layer into Vector<Neurons> to feed into Layer constructor
    vector<Neuron> oneDimage(numElements);
    for (double i = 0; i < numElements; ++i) 
    {
        double pixelValue = imageLayer[i];     
        Neuron n(pixelValue);
        oneDimage[i] = n;
    }
    
    int Rows = 28;
    int Col = 28;
    double bias = 1;
    Layer ImageLayer(Rows, Col, oneDimage, filter, bias); 

    ImageLayer.printLayerValues();
    cout<< endl;

    Layer convolvedLayer = ImageLayer.Convolution(ImageLayer, filter);

    convolvedLayer.printLayerValues();
    cout << endl;

    Layer PooledLayer = convolvedLayer.Pooling(convolvedLayer, 2);

    PooledLayer.printLayerValues();
    cout << endl;

    Layer convolvedLayer2 = PooledLayer.Convolution(PooledLayer, filter);
    
   convolvedLayer2.printLayerValues();
    cout << endl;

    Layer PooledLayer2 = convolvedLayer2.Pooling(convolvedLayer2, 2);

   PooledLayer2.printLayerValues();
    cout << endl;

    vector<Neuron> flattened = PooledLayer2.flatten(PooledLayer2);

    Layer FullyConnected120 = PooledLayer2.FullyConnected(flattened, 120);

    FullyConnected120.printLayerValues();
    cout << endl;

    
    
  






    







}
