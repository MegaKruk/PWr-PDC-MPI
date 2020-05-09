#include <mpi.h>
#include <stdio.h>

#define SIZE 10

int matrix1[SIZE][SIZE];
int matrix2[SIZE][SIZE];

typedef struct 
{
    // number of processes
    int worldSize;
    // rank of the process
    int worldRank;
    // Get the name of the processor
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    int column;
    int row;
    int dimensions;
    MPI_Comm commGrid;
    MPI_Comm commRow;
    MPI_Comm commCol;
} mpiGrid;

int main(int argc, char** argv) 
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    mpiGrid grid;

    MPI_Comm_size(MPI_COMM_WORLD, &(grid.worldSize));
    MPI_Comm_rank(MPI_COMM_WORLD, &(grid.worldRank));
    MPI_Get_processor_name(grid.processorName, &(grid.nameLen));

    // debug cout
    std::cout << "Hello world from processor " << grid.processorName << ", rank " << grid.worldRank 
              << " out of " << grid.worldSize << " processors\n";
              //<< "\ncommGrid: " << grid.commGrid << "\ncommRow: "<< grid.commRow << "\ncommCol: " << grid.commCol << "\n" << std::endl;

    // Finalize the MPI environment.
    MPI_Finalize();
}