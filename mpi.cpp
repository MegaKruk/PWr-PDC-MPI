#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>

#define N 20
/*
Allowed number of processes for mpirun "p" is such that sqrt(p) must be an integer
Matrix size N must be dividable by sqrt(p)
*/

int matrix1[N][N];
int matrix2[N][N];
int matrix3[N][N];
std::chrono::high_resolution_clock::time_point start;
std::chrono::high_resolution_clock::time_point finish;

typedef struct 
{
    // number of processes
    int worldSize;
    // rank of the process
    int worldRank;
    // name of the processor
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    
    int column;
    int row;
    int dimensions;
    
    MPI_Comm commGrid;
    MPI_Comm commRow;
    MPI_Comm commCol;
} mpiGrid;

void timeStart()
{
    start = std::chrono::high_resolution_clock::now();
}

long double timeStop()
{
    std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<long double, std::nano> diff = finish - start;
    return diff.count();
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

int checkSize(int n, int processes)
{
    int check1 = sqrt(processes);
    int check2 = n / check1;
    if(check1 * check2 != n)
    {
        MPI_Finalize();
        return 1;
    }
    return 0;
}

// just for now, later will be loading data from ciphertext/plaintext and coding/decoding into matrix
void matrixInit() 
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) 
        {
            matrix1[i][j] = rand() % 10;
            matrix2[i][j] = rand() % 10;
        }
    }
}

void matrixMultiply(int n) 
{
    int tmp = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) 
        {
            tmp = 0;
            for (int k = 0; k < n; k++)
                tmp += (matrix1[i][k] * matrix2[k][j]);
            matrix3[i][j] += tmp;
        }
    }
}

void matrixPrint(int n, int *matrix)
{
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            std::cout << matrix[i * n + j];
            std::cout << "\t";
        }
        std::cout << std::endl;
    }
}

void gridInit(mpiGrid *grid) 
{
    int rank;
    int wrap[2];
    int dims[2];
    int coords[2];
    int freeCoords[2];

    MPI_Comm_size(MPI_COMM_WORLD, &(grid->worldSize));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(grid->processorName, &(grid->nameLen));
    
    grid->dimensions = (int) sqrt((double) grid->worldSize); //!!!
    dims[0] = dims[1] = grid->dimensions;
    wrap[0] = 0;
    wrap[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, wrap, 1, &(grid->commGrid));
    MPI_Comm_rank(grid->commGrid, &(grid->worldRank));
    MPI_Cart_coords(grid->commGrid, grid->worldRank, 2, coords);
    
    grid->row = coords[0];
    grid->column = coords[1];
    freeCoords[0] = 0;
    freeCoords[1] = 1;

    MPI_Cart_sub(grid->commGrid, freeCoords, &(grid->commRow));
    
    freeCoords[0] = 1;
    freeCoords[1] = 0;

    MPI_Cart_sub(grid->commGrid, freeCoords, &(grid->commCol));

    // debug cout
    //std::cout << "\nHello world from processor " << grid.processorName << ", rank " << grid.worldRank 
              //<< " out of " << grid.worldSize << " processors" << std::endl;
              //<< "\ncommGrid: " << grid.commGrid << "\ncommRow: "<< grid.commRow << "\ncommCol: " << grid.commCol << "\n" << std::endl;
}

void sanityCheck(int n, mpiGrid *grid)
{
    // master instructions
    if(grid->worldRank == 0)
    {
        //std::cout << "\nI'm master!\n";
        int code1 = checkProcesses(grid->worldSize);
        if(code1 == 1)
        {
            std::cout << "\nWrong number of processes!\n";
            exit(1);
        }
        int code2 = checkSize(n, grid->worldSize);
        if(code2 == 1)
        {
            std::cout << "\nWrong matrix size!\n";
            exit(2);
        }

        std::cout << "\nMatrix 1:" << std::endl;
        matrixPrint(n, *matrix1);
        std::cout << "\nMatrix 2:" << std::endl;
        matrixPrint(n, *matrix2);

        //local test!!!
        /*
        timeStart();
        matrixMultiply(n);
        long double duration = timeStop();
        std::cout << "\nMatrix 3: " << std::endl;
        matrixPrint(n, *matrix3);
        std::cout << "\nDuration: " << duration << " ns" << std::endl;
        */
    }
    else
    // slaves instructions
    {
        //std::cout << "\nI'm slave!\n"
        // just one process needs to call exit fun
        /*int code = checkProcesses(grid.worldSize);
        if(code == 1)
            exit(1);*/
    }
}

void bufferToMatrix(int *buff, int a[][N], int n) 
{
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) 
        {
            a[i][j] = buff[k];
            k++;
        }
    }
}

void matrixToBuffer(int *buff, int a[][N], int n) 
{
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) 
        {
            buff[k] = a[i][j];
            k++;
        }
    }
}

void foxAlgorithm(int n, mpiGrid *grid) 
{
    int tmpMatrix[N][N];
    int *buff; 
    int stage, root, subDimension, src, dst;
    MPI_Status status;

    subDimension = n / grid->dimensions;


    buff = new int[subDimension * subDimension];
    for (int i = 0; i < subDimension * subDimension; i++)
        buff[i] = 0;

    src = (grid->row + 1) % grid->dimensions;
    dst = (grid->row + grid->dimensions - 1) % grid->dimensions;

    for (stage = 0; stage < grid->dimensions; stage++) 
    {
        root = (grid->row + stage) % grid->dimensions;
        if (root == grid->column) 
        {
            matrixToBuffer(buff, matrix1, subDimension);
            MPI_Bcast(buff, subDimension * subDimension, MPI_INT, root, grid->commRow);
            bufferToMatrix(buff, matrix1, subDimension);
            matrixMultiply(subDimension);
        } 
        else 
        {
            matrixToBuffer(buff, tmpMatrix, subDimension);
            MPI_Bcast(buff, subDimension * subDimension, MPI_INT, root, grid->commRow);
            bufferToMatrix(buff, tmpMatrix, subDimension);
            matrixMultiply(subDimension);
        }
        matrixToBuffer(buff, matrix2, subDimension);
        MPI_Sendrecv_replace(buff, subDimension * subDimension, MPI_INT, dst, 0, src, 0, grid->commCol, &status);
        bufferToMatrix(buff, matrix2, subDimension);
    }
}

void measureStuff(int n, mpiGrid *grid)
{
    int checkboardFieldSize = n / grid->dimensions;
    MPI_Barrier(grid->commGrid);
    if (grid->worldRank == 0)
    {
        timeStart();
    }
    foxAlgorithm(n, grid);
    MPI_Barrier(grid->commGrid);
    if (grid->worldRank == 0)
    {
        long double duration = timeStop();
        std::cout << "\nDuration: " << duration << " ns" << std::endl;
    }

    int *resultBuff = new int[n * n];
    int *localBuff = new int[checkboardFieldSize * checkboardFieldSize];
    matrixToBuffer(localBuff, matrix3, checkboardFieldSize);

    MPI_Gather(localBuff, checkboardFieldSize * checkboardFieldSize, MPI_INT, resultBuff, checkboardFieldSize * checkboardFieldSize, MPI_INT, 0, grid->commGrid);
    MPI_Barrier(grid->commGrid);
    if (grid->worldRank == 0) 
    {
        int *result = new int[n * n];
        int k = 0;
        for (int l = 0; l < grid->dimensions; l++)
        {
            for (int m = 0; m < grid->dimensions; m++)
            {
                for (int i = l * checkboardFieldSize; i < l * checkboardFieldSize + checkboardFieldSize; i++)
                {
                    for (int j = m * checkboardFieldSize; j < m * checkboardFieldSize + checkboardFieldSize; j++) 
                    {
                        result[i * n + j] = resultBuff[k];
                        k++;
                    }
                }
            }
        }
        // print result
        std::cout << "\nResult: " << std::endl;
        matrixPrint(n, result);
    }
}

int main(int argc, char** argv) 
{
    srand(time(NULL));
    
    matrixInit();

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Initialize grid
    mpiGrid grid;
    gridInit(&grid);

    // check if everything is correct
    sanityCheck(N, &grid);

    // main function
    measureStuff(N, &grid);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}