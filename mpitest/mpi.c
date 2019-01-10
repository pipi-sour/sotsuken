#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, proc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    printf("%d of %d\n", rank, proc);
    MPI_Finalize();
    return 0;
}