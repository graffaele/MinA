#include "MinA/algorithm/SimplexParallel.h"
using namespace std;

SimplexParallel::SimplexParallel(int stop) : Simplex(stop)
{
    if (!MPI::Is_initialized())
        throw runtime_error("MPI_INIT must be used beforehand");
}

SimplexParallel::~SimplexParallel() {}

Result SimplexParallel::algorithm(shared_ptr<FunctionToBeOptimized> start)
{

    mFunction = start;
    mDimension = mFunction->getParSpaceDim();
    verticesVector A;

    // restore();
    // cout << "Start with current loop= " << currentIteration << endl;
    setStepSize();
    if (Acopy.empty()) {
        A.resize(mDimension + 1);
        initializeVertices(A); // Initial
    }
    else {
        A = Acopy;
    }

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        int mode;

        while (checkStoppingCondition()) {

            for (int i = 0; i <= mDimension; i++) {
                A[i].second = mFunction->evaluate(A[i].first);
            } // Evaluate(A)
            sort(A.begin(), A.end(),
                 [](vertex& a, vertex& b) -> bool { return a.second < b.second; });

            ofstream verticesFile;
            string outFile_vertices("parallelSimplex_" + getFunctionName() + "_" +
                                    to_string(world_size) + "_Vertices");
            verticesFile.open(outFile_vertices, ios::app);
            verticesFile << "  Iteration: " << currentIteration << endl;
            printOutVertices(A, verticesFile);
            verticesFile << endl;
            verticesFile.close();

            // Centroid of the simplex, excluding world_size vertices
            vertex M = getCentroid(A, world_size);
            M.second = mFunction->evaluate(M.first);

            ofstream meanFile;
            string outFile_mean("parallelSimplex_" + getFunctionName() + "_" +
                                to_string(world_size) + "_Size");
            meanFile.open(outFile_mean, ios::app);
            meanFile << "Iteration= " << currentIteration << "	Mean= " << M.second
                     << "	SimplexSize =	" << getSimplexSize(A) << "	";
            meanFile << endl;
            meanFile.close();
            mode = 1;
            for (int i = 1; i < world_size; i++) {
                MPI_Send(&mode, 1, MPI_INT, i, world_size, MPI_COMM_WORLD);
                sendVertex(M, i, 0);
                sendVertex(A[0], i, 1);
                sendVertex(A[mDimension - world_size + i], i, 2);
                sendVertex(A[mDimension - world_size + i + 1], i, 3);

            } // sending M,A0,Aj_1,Aj

            int sumcheck = 0;

            for (int i = 1; i < world_size; i++) {
                int check;
                MPI_Recv(&(check), 1, MPI_INT, i, i + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                sumcheck += check;
            }
            if (sumcheck == 0) {
                for (int i = 1; i < mDimension - world_size + 1; i++) {
                    A[i] = getShrinkedPoint(A[i], A[0]);
                }
                for (int i = 1; i < world_size; i++) {
                    vertex b = receiveVertex(i, i);
                    A[mDimension - world_size + i + 1] = getShrinkedPoint(b, A[0]);
                }
            } // case 4
            else
                for (int i = 1; i < world_size; i++) {
                    vertex b = receiveVertex(i, i);
                    b.second = mFunction->evaluate(b.first);
                    A[mDimension - world_size + i + 1] = b;
                }

            ofstream fValueFile;
            string outFile_fValue("parallelSimplex_" + getFunctionName() + "_" +
                                  to_string(world_size) + "_fValue");
            fValueFile.open(outFile_fValue, ios::app);
            fValueFile << "Iteration= " << currentIteration << "   ";
            printOutVertex(A[0], "A[0]", fValueFile);
            fValueFile.close();
            // save();
        }

        mode = 0;
        for (int i = 1; i < world_size; i++) {
            MPI_Send(&mode, 1, MPI_INT, i, world_size, MPI_COMM_WORLD);
        }
    }

    if (world_rank != 0) {

        while (1) {
            int mode;
            MPI_Recv(&mode, 1, MPI_INT, 0, world_size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (mode == 1) {
                vertex M, Aj, Aj_1, A0, Ar, Ac, Ae, Ap;
                // receiving M,A0,Aj,Aj_1
                M = receiveVertex(0, 0);
                A0 = receiveVertex(0, 1);
                Aj_1 = receiveVertex(0, 2);
                Aj = receiveVertex(0, 3);
                A0.second = mFunction->evaluate(A0.first);
                Aj_1.second = mFunction->evaluate(Aj_1.first);
                Aj.second = mFunction->evaluate(Aj.first);
                int check = 0;

                // Update vertex step3
                Ar = getReflectedPoint(M, Aj);
                if (Ar.second < A0.second) {
                    Ae = getExtendedPoint(M, Ar);
                    if (Ae.second < A0.second) {
                        Aj = Ae;
                        check = 1;
                    }
                    else {
                        Aj = Ar;
                        check = 2;
                    }
                } // case 1
                else if (Ar.second < Aj_1.second) {
                    Aj = Ar;
                    check = 3;
                } // case 2
                else {
                    if (Ar.second < Aj.second) {
                        Ap = Ar;
                    }
                    else {
                        Ap = Aj;
                    } // Ac
                    Ac = getContractedPoint(M, Ap);

                    if (Ac.second < Ap.second) {
                        // if(Ac.second<Aj.second)cout<<"Ac="<<Ac.second<<endl;
                        // else 	cout<<"Aj="<<Aj.second<<endl;
                        Aj = Ac;
                        check = 4;
                    } // case 3
                    else
                        Aj = Ap;
                }

                sendVertex(Aj, 0, world_rank);
                MPI_Send(&(check), 1, MPI_INT, 0, world_rank + 1, MPI_COMM_WORLD);
            }
            else if (mode == 0) {
                Result rs;
                return rs;
            }
        }
    }

    if (world_rank == 0) {

        Result rs;
        pushResult(rs, A[0]); // Result
        return rs;
    }
    Result rs;
    return rs;
}

void SimplexParallel::sendVertex(vertex& An, int receiver, int tag)
{
    double Ad[mDimension];
    for (int iPar = 0; iPar < mDimension; ++iPar)
        Ad[iPar] = An.first[iPar];

    MPI_Send(&(Ad[0]), mDimension, MPI_DOUBLE, receiver, tag, MPI_COMM_WORLD);
}

vertex SimplexParallel::receiveVertex(int sender, int tag)
{
    vertex An;
    An.first.resize(mDimension);
    double Ad[mDimension];
    MPI_Recv(&(Ad[0]), mDimension, MPI_DOUBLE, sender, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int iPar = 0; iPar < mDimension; ++iPar)
        An.first[iPar] = Ad[iPar];

    return An;
}

vertex SimplexParallel::getContractedPoint(vertex& M, vertex& Ajp)
{
    // Return contraction point Ac of initial point Ajp w.r.t. centroid M
    // Ac = beta * (Ajp + M)
    //
    vertex Ac;
    Ac.first.resize(mDimension);
    for (int iPar = 0; iPar < mDimension; ++iPar)
        Ac.first[iPar] = getMetaParameter("beta") * (Ajp.first[iPar] + M.first[iPar]);
    Ac.second = mFunction->evaluate(Ac.first);
    return Ac;
}
