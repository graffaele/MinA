#ifndef SIMPLEX_H
#define SIMPLEX_H
#include <map>
#include <iostream>
#include <memory>
#include "MinA/common/OptimizationAlgorithm.h"
#include "MinA/common/Log.h"
#include <boost/serialization/utility.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <string>
#include <fstream>
#include <cstdio>

typedef std::pair<std::vector<double>, double> vertex;
typedef std::vector<vertex> verticesVector;
class Simplex : public OptimizationAlgorithm {
  public:
    Simplex(int stop = 100);
    ~Simplex();
    virtual Result algorithm(std::shared_ptr<FunctionToBeOptimized> start);
    void printOutVertices(verticesVector& simplexVertices, std::ostream& outStream = std::cout);
    void printOutVertex(vertex& simplexVertex, std::string vertexName,
                        std::ostream& outStream = std::cout);
    void checkBoundaryCondition(vertex& A);
    void setStepSize(std::vector<double> s);
    void setStepSize();
    void setStoppingIteration(int n);
    void setFunctionName(std::string name);
    std::string getFunctionName();
    double getSimplexSize(verticesVector& para);

  protected:
    int mDimension;
    std::shared_ptr<FunctionToBeOptimized> mFunction;
    std::vector<double> stepSize;
    bool checkStoppingCondition();
    void pushResult(Result& rs, vertex& A);
    void initializeVertices(verticesVector& A);
    vertex getCentroid(verticesVector& A, int nExcluded);
    vertex getReflectedPoint(vertex& M, vertex& Aj);
    vertex getExtendedPoint(vertex& M, vertex& Ar);
    virtual vertex getContractedPoint(vertex& M, vertex& Ajp);
    vertex getShrinkedPoint(vertex& Ap, vertex& A0);
    void save() const;
    void restore();
    verticesVector Acopy;
    int currentIteration;
    double difference;
    int stoppingIteration;
    std::string FunctionName;
};

#endif
