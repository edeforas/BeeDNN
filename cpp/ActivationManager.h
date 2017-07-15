#ifndef ActivationManager_
#define ActivationManager_

#include <string>
#include <vector>
using namespace std;

class Activation;

class ActivationManager
{
public:
    ActivationManager();
    virtual ~ActivationManager();

    Activation* get_activation(string name); //do not delete: manager own it.

    void list_all(vector<string>& allActivationNames);

private:
    vector<Activation*> _vActivations;
};

#endif
