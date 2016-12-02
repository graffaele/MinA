#include <iostream>
#include <map>
#include "MinA/common/Minimizer.h"
#include "MinA/common/Result.h"
#include "myfunction.cpp"
#include "MinA/algorithm/Simplex.h"
#include <utility>
using namespace std;

int main(void)
{
    // badFunction.setHandle(KILL);
    vector<Parameter> params;
    params.push_back({ "x", 3 });
    params.push_back({ "y", 4 });
    params.push_back({ "z", 5 });
    shared_ptr<BadTen> badFunction(new BadTen(params));

    shared_ptr<Simplex> serialSimplex(new Simplex);
    Minimizer mina;
    int count = 0;
    for (int i = 0; i < 10; i++) {
        try {
            mina.minimize(badFunction, serialSimplex);
        }
        catch (Arithmetical_Exception& ae) {
            count++;
        }
    }
    cout << "counter = " << count << endl;
    return !(count == 10);
}
