#include <iostream>
using namespace std;

#include "BeeDNNLib.h"

#include "Activations.h"

void hello()
{
	cout << "Hello" << endl;

}

float activation(char * activ,float f)
{
	Activation * pactiv = get_activation(activ);
	float f2=pactiv->apply(f);
	delete pactiv;
	return f2;
}

