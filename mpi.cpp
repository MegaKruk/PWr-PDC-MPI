#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <math.h>
#include <chrono>
#include <string>
#include <sstream>

#define N 1000

using namespace std;
/*  
Allowed number of processes for mpirun "p" is such that sqrt(p) must be an integer  
Matrix size N must be dividable by sqrt(p)  
*/

int matrix1[N][N];
int matrix2[N][N];
chrono::high_resolution_clock::time_point start;
chrono::high_resolution_clock::time_point finish;
chrono::high_resolution_clock::time_point beginning;
chrono::high_resolution_clock::time_point end;

typedef struct 
{
    int worldRank;
    int dim;
    int row;
    int column;
    int rank;

    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;

    MPI_Comm commGrid;
    MPI_Comm commRow;
    MPI_Comm commcolumn;
} mpiGrid;

void timeStart()
{
    start = chrono::high_resolution_clock::now();
}

long double timeStop()
{
    chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
    chrono::duration<long double, nano> diff = finish - start;
    return diff.count();
}

void totalElapsedStart()
{
    beginning = chrono::high_resolution_clock::now();
}

void totalElapsedCount()
{
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    chrono::duration<long double, nano> diff2 = end - beginning;
    long double totalElapsed = diff2.count();
    cout << "\nTotal duration: " << totalElapsed << " ns" << endl;
    ofstream out3("times.csv", fstream::app);
    out3 << "Total duration:\t" << totalElapsed << "\tns\n";
    out3.close();
}

int checkProcesses(int processes)
{
    int check = sqrt(processes);
    if((check * check) != processes) 
    {
        MPI_Finalize();
        return 1;
    }
    return 0;
}

int checkSize(int processes)
{
    int check1 = sqrt(processes);
    int check2 = N / check1;
    if(check1 * check2 != N)
    {
        MPI_Finalize();
        return 1;
    }
    return 0;
}

void gridInit(mpiGrid *grid) 
{
    int dims[2];
    int wrap[2];
    int coords[2];
    int freeCoords[2];
    int worldRank;

    MPI_Comm_size(MPI_COMM_WORLD, &(grid->worldRank));
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    grid->dim = (int) sqrt((double) grid->worldRank);
    dims[0] = dims[1] = grid->dim;

    wrap[0] = 0;
    wrap[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, wrap, 1, &(grid->commGrid));
    MPI_Comm_rank(grid->commGrid, &(grid->rank));
    MPI_Cart_coords(grid->commGrid, grid->rank, 2, coords);
    grid->row = coords[0];
    grid->column = coords[1];

    freeCoords[0] = 0;
    freeCoords[1] = 1;
    MPI_Cart_sub(grid->commGrid, freeCoords, &(grid->commRow));

    freeCoords[0] = 1;
    freeCoords[1] = 0;
    MPI_Cart_sub(grid->commGrid, freeCoords, &(grid->commcolumn));
}


void matrixMultiply(int **a, int **b, int **c, int size) 
{
    int temp = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) 
        {
            temp = 0;
            for (int k = 0; k < size; k++)
                temp += (a[i][k] * b[k][j]);
            c[i][j] += temp;
        }
    }
}

void bufferToMatrix(int *buff, int **a, int size) 
{
    int k = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) 
        {
            a[i][j] = buff[k];
            k++;
        }
    }
}

void matrixToBuffer(int *buff, int **a, int size) 
{
    int k = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++) 
        {
            buff[k] = a[i][j];
            k++;
        }
    }
}

void matrixPrint(int **matrix, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            int el = matrix[i][j];
            cout << el;
            cout << "\t";
        }
        cout << endl;
    }
}

void bufferPrint(int *matrix, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            int el = matrix[i * size + j];
            cout << el;
            cout << "\t";
        }
        cout << endl;
    }
}

void matrixInit() 
{
    ifstream in;
    in.open("matrix1_1000.csv");
    for (int i = 0; i < N; i++)
    {
        string line;
        getline(in, line);
        stringstream iss(line);
        for (int j = 0; j < N; j++) 
        {
            string val;
            getline(iss, val, ',');
            stringstream convertor(val);
            convertor >> matrix1[i][j];
        }
    }
    in.close();

    ifstream in2;
    in2.open("matrix2_1000.csv");
    for (int i = 0; i < N; i++)
    {
        string line2;
        getline(in2, line2);
        stringstream iss2(line2);
        for (int j = 0; j < N; j++) 
        {
            string val2;
            getline(iss2, val2, ',');
            stringstream convertor2(val2);
            convertor2 >> matrix2[i][j];
        }
    }
    in2.close();

}

void foxAlgorithm(int n, mpiGrid *grid, int **a, int **b, int **c) 
{
    int **tempMatrix, *buff, stage, root, subDimension, src, dst;
    MPI_Status status;

    subDimension = n / grid->dim;

    tempMatrix = new int*[subDimension];
    for(int i = 0; i < subDimension; ++i)
        tempMatrix[i] = new int[subDimension];
    for (int i = 0; i < subDimension; i++)
        for (int j = 0; j < subDimension; j++)
            tempMatrix[i][j] = 0;

    buff = new int[subDimension*subDimension];
    for (int i = 0; i < subDimension * subDimension; i++)
        buff[i] = 0;

    src = (grid->row + 1) % grid->dim;
    dst = (grid->row + grid->dim - 1) % grid->dim;

    for (stage = 0; stage < grid->dim; stage++) 
    {
        root = (grid->row + stage) % grid->dim;
        if (root == grid->column) 
        {
            matrixToBuffer(buff, a, subDimension);
            MPI_Bcast(buff, subDimension * subDimension, MPI_INT, root, grid->commRow);
            bufferToMatrix(buff, a, subDimension);
            matrixMultiply(a, b, c, subDimension);
        } 
        else 
        {
            matrixToBuffer(buff, tempMatrix, subDimension);
            MPI_Bcast(buff, subDimension * subDimension, MPI_INT, root, grid->commRow);
            bufferToMatrix(buff, tempMatrix, subDimension);
            matrixMultiply(tempMatrix, b, c, subDimension);
        }
        matrixToBuffer(buff, b, subDimension);
        MPI_Sendrecv_replace(buff, subDimension * subDimension, MPI_INT, dst, 0, src, 0, grid->commcolumn, &status);
        bufferToMatrix(buff, b, subDimension);
    }
}

void sanityCheck(mpiGrid *grid)
{
    // master instructions
    if(grid->worldRank == 0)
    {
        //cout << "\nI'm master!\n";
        int code1 = checkProcesses(grid->worldRank);
        if(code1 == 1)
        {
            cout << "\nWrong number of processes!\n";
            exit(1);
        }
        int code2 = checkSize(grid->worldRank);
        if(code2 == 1)
        {
            cout << "\nWrong matrix size!\n";
            exit(2);
        }

        cout << "\nMatrix 1:" << endl;
        bufferPrint(*matrix1, N);
        cout << "\nMatrix 2:" << endl;
        bufferPrint(*matrix2, N);
    }
    else
    // slaves instructions
    {
        //cout << "\nI'm slave!\n"
    }
}

int** testStuff()
{
    int **testResult, **testMatrix1, **testMatrix2;
    testMatrix1 = new int*[N];
    testMatrix2 = new int*[N];
    testResult = new int*[N];
    for(int i = 0; i < N; ++i) 
    {
        testMatrix1[i] = new int[N];
        testMatrix2[i] = new int[N];
        testResult[i] = new int[N];
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) 
        {
            testMatrix1[i][j] = matrix1[i][j];
            testMatrix2[i][j] = matrix2[i][j];
            testResult[i][j] = 0;
        }
    matrixMultiply(testMatrix1, testMatrix2, testResult, N);
    return testResult;
}

void measureStuff(mpiGrid *grid)
{
    int **matrixA, **matrixB, **matrixC;
    int checkboardPiece = N / grid->dim;
    int checkboardRow = grid->row * checkboardPiece;
    int checkboardColumn = grid->column * checkboardPiece;

    matrixA = new int*[N];
    matrixB = new int*[N];
    matrixC = new int*[N];

    for(int i = 0; i < N; ++i)
    {
        matrixA[i] = new int[N];
        matrixB[i] = new int[N];
        matrixC[i] = new int[N];
    }

    for (int i = checkboardRow; i < checkboardRow + checkboardPiece; i++)
    {
        for (int j = checkboardColumn; j < checkboardColumn + checkboardPiece; j++) 
        {
            matrixA[i - (checkboardRow)][j - (checkboardColumn)] = matrix1[i][j];
            matrixB[i - (checkboardRow)][j - (checkboardColumn)] = matrix2[i][j];
            matrixC[i - (checkboardRow)][j - (checkboardColumn)] = 0;
        }
    }

    if (grid->rank == 0)
    {
        totalElapsedStart();
        cout << "\nMatrix 1:" << endl;
        bufferPrint(*matrix1, N);
        cout << "\nMatrix 2:" << endl;
        bufferPrint(*matrix2, N);
    }

    MPI_Barrier(grid->commGrid);
    if (grid->rank == 0)
    {
        timeStart();
    }
    foxAlgorithm(N, grid, matrixA, matrixB, matrixC);
    MPI_Barrier(grid->commGrid);
    if (grid->rank == 0)
    {
        long double duration = timeStop();
        cout << "\nComputation duration: " << duration << " ns" << endl;
        ofstream out("times.csv", fstream::app);
        out << "Computation duration:\t" << duration << "\tns\n";
        out.close();
    }

    int *resultsBuffer = new int[N * N];
    int *tmpBuffer = new int[checkboardPiece * checkboardPiece];
    matrixToBuffer(tmpBuffer, matrixC, checkboardPiece);

    MPI_Gather(tmpBuffer, checkboardPiece * checkboardPiece, MPI_INT, resultsBuffer, checkboardPiece * checkboardPiece, MPI_INT, 0, grid->commGrid);
    MPI_Barrier(grid->commGrid);
    if (grid->rank == 0) 
    {
        int *results = new int[N * N];
        int k = 0;
        for (int x = 0; x < grid->dim; x++)
        {
            for (int z = 0; z < grid->dim; z++)
                for (int i = x * checkboardPiece; i < x * checkboardPiece + checkboardPiece; i++)
                    for (int j = z * checkboardPiece; j < z * checkboardPiece + checkboardPiece; j++) 
                    {
                        results[i * N + j] = resultsBuffer[k];
                        k++;
                    }
        }

        // test result
        cout << "\nFox's result:\n";
        bufferPrint(results, N);
        int** res = testStuff();
        cout << "\nLocal test:\n";
        matrixPrint(res, N);
        for(int i = 0;i < N; i++)
            for(int j = 0; j < N; j++)
            {
                if(res[i][j] != results[i * N + j])
                {
                    cout << "\nFox's solution wrong!" << endl;
                    exit(1);
                }
            }
        cout << "\nFox's solution ok" << endl;

        // save result
        ofstream out("out.csv");
        for(int i = 0; i < N; i++) 
        {
            for (int j = 0; j < N; j++)
                out << results[i * N + j] <<',';
            out << '\n';
        }
        out.close();
        totalElapsedCount();
    }
}

int main(int argc, char *argv[]) 
{
    
    //srand(time(NULL));

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Initialize grid and then matrix
    mpiGrid grid;
    gridInit(&grid);
    matrixInit();

    // check if everything is correct
    sanityCheck(&grid);

    // main function
    measureStuff(&grid);    

    // Finalize the MPI environment
    MPI_Finalize();
    
    exit(0);
}       
