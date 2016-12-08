#include <iostream>
#include <map>
#include "mpi.h"
#include "MinA/common/Minimizer.h"
#include "MinA/common/Result.h"
#include "TestFunction.cpp"
#include "MinA/algorithm/Simplex.h"
#include "MinA/algorithm/SimplexParallel.h"
#include <utility>
using namespace std;

int main(int argc, char** argv)
{

    if (!MPI::Is_initialized())
        MPI::Init();
    Minimizer mina;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    shared_ptr<Simplex> nmSimplex(new Simplex(10));
    nmSimplex->setFunctionName("Square");
    // shared_ptr<SimplexParallel> parallelSimplex(new SimplexParallel(1000));
    shared_ptr<SimplexParallel> parallelSimplex(new SimplexParallel(10));
    parallelSimplex->setFunctionName("Square");
    parallelSimplex->setMetaParameter("beta", 0.2);
    parallelSimplex->setMetaParameter("tau", 0.2);

    vector<string> parNames;
    for (int i = 0; i < 10; ++i)
        parNames.push_back(string("x" + to_string(i)));
    shared_ptr<SquareFunction> fSquare(new SquareFunction(parNames, 8.0, -10.0, 10.0));
    shared_ptr<Gaussian_function> fGauss(new Gaussian_function(parNames, 5., -10., 10.));

    parNames.resize(0);
    parNames.push_back("x");
    parNames.push_back("y");
    shared_ptr<McCormick_function> fMcCormick(
      new McCormick_function(parNames, { 0., 0. }, { -1.5, 3.0 }, { 4., 4. }));
    shared_ptr<Boot_s_function> fBoot_s(
      new Boot_s_function(parNames, { 3., 4. }, { -10., -10.0 }, { 10., 10. }));
    shared_ptr<Michalewicz_function> fMichalewicz(
      new Michalewicz_function(parNames, { 3., 4. }, { -10., -10.0 }, { 10., 10. }));
    shared_ptr<Himmelblaufunction> fHimmelblau(
      new Himmelblaufunction(parNames, { 3., 4. }, { -10., -10 }, { 10., 10. }));
    shared_ptr<Schwefel_function> fSchwefel(new Schwefel_function({ "x" }, 50., -500, 500.));
    /*
    map<string, double> input;
    double s,e,r;
    s=-5;
    e=5;
    r=0.05;

     ofstream data;
     data.open("z_test_function_plot", ios::app);
     for(double i=s;i<=e;i+=0.01){
        input["x0"]=i;
        data<<" "<<i<<" "<<fSquare->getEvaluation(input)<<endl;

    }
    data.close();*/

    if (world_rank == 0)
        cout << "function minimized value=" << mina.minimize(fGauss, parallelSimplex).result << endl;
    else
        auto rs = mina.minimize(fGauss, parallelSimplex);

    MPI_Finalize();
    return (0);
}
