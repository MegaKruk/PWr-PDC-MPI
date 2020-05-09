#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define N 10
/*
Allowed number of processes for mpirun p is sych that sqrt(p) must be an integer
Matrix size N must be dividable by sqrt(p)
*/

int matrix1[N][N];
int matrix2[N][N];

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

int checkProcesses(int processes)
{
    int check = sqrt(processes);
    if ((check * check) != processes) 
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
            matrix1[i][j] = rand() % 29;
            matrix2[i][j] = rand() % 29;
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

int main(int argc, char** argv) 
{
    matrixInit();

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    mpiGrid grid;

    MPI_Comm_size(MPI_COMM_WORLD, &(grid.worldSize));
    MPI_Comm_rank(MPI_COMM_WORLD, &(grid.worldRank));
    MPI_Get_processor_name(grid.processorName, &(grid.nameLen));

    // common instructions
    std::cout << "\nHello world from processor " << grid.processorName << ", rank " << grid.worldRank 
              << " out of " << grid.worldSize << " processors" << std::endl;
              //<< "\ncommGrid: " << grid.commGrid << "\ncommRow: "<< grid.commRow << "\ncommCol: " << grid.commCol << "\n" << std::endl;

    // master instructions
    if(grid.worldRank == 0)
    {
        std::cout << "\nI'm master!\n";
        int code = checkProcesses(grid.worldSize);
        if(code == 1)
        {
            std::cout << "\nWrong number of processes!\n";
            exit(1);
        }

        std::cout << "\nMatrix 1:" << std::endl;
        matrixPrint(N, *matrix1);
        std::cout << "\nMatrix 2:" << std::endl;
        matrixPrint(N, *matrix2);
        std::cout << std::endl;

    }
    else
    // slaves instructions
    {
        int code = checkProcesses(grid.worldSize);
        if(code == 1)
            exit(1);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
}