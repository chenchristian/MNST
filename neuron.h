#ifndef NEURON.H
#define NEURON.H


#include <iostream>

using namespace std;

class Neuron {
public:
    Neuron();
    Neuron(double pixels);
    Neuron(double pixels, double weightInput);
    double getValue() const;
    double getWeight() const;
    

    
   
private: 
    double value;
    double weight;
};







#endif
