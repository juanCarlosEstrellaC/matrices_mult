#include <memory>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>

#define MATRIX_DIM 25

void imprimir_vector(const double* vector, int size)
{
    printf("{ ");
    for (int i = 0; i < size; ++i)
    {
        printf("%.0f%s", vector[i], (i < size - 1 ? ", " : " "));
    }
    printf("}\n");
}

void imprimir_matriz(double* A, int rows, int cols)
{
    // A es realmente un vector 1D, pero se imprime como una matriz.
    for (int i = 0; i < rows; i++)
    {
        printf("i: %d,   ", i);
        for (int j = 0; j < cols; j++)
        {
            int index = i * cols + j;
            printf("%.0f ", A[index]);
        }
        printf("\n");
    }
    printf("\n");
}

void multiplicar_matriz(double* A, double* x, double* b, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        b[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            int index = i * cols + j;
            b[i] += A[index] * x[j];
        }
    }
}

/* Tomar en cuenta que la matriz A es en un vector 1D, ie, A[MATRIX_DIM * MATRIX_DIM]
 * En mi forma de hacer, no aumento el tamaño de la matriz, sino que divido la matriz en bloques iguales mientras
 * sea posible, y el último proceso se queda con el resto. Por ello no ocupo el ceil.
 * A*x = b
 * A(MATRIX_DIM x MATRIX_DIM) * x(MATRIX_DIM x 1) = b(MATRIX_DIM x 1)
 * A(25 x 25) * x(25 x 1) = b(25 x 1)
 *
 * Al no usar el ceil ni double, la división no queda como lo visto en clase, ie, filas_por_rank[4] = {7, 7, 7, 4};, sino:
 *  filas_por_rank: 6 6 6 7
    bloques1: 150 150 150 175
    desplazamientos1: 0 150 300 450
    desplazamientos_b: 0 6 12 18
 */
int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Dividir la matriz en bloques iguales (o casi iguales) para distribuirlos entre todos los procesos.
    int e = MATRIX_DIM / nprocs;
    int f = MATRIX_DIM - e * (nprocs - 1);

    std::vector<int> filas_por_rank(nprocs);     //filas_por_rank[4] = {6, 6, 6, 7};
    for (int i = 0; i < nprocs; i++){
        filas_por_rank[i] = (i == nprocs - 1) ? f : e;
    }

    std::vector<int> elementos_por_rank_A(nprocs);     //elementos_por_rank_A[4] = {6 * 25, 6 * 25, 6 * 25, 7 * 25};
    for (int i = 0; i < nprocs; i++){
        elementos_por_rank_A[i] = (filas_por_rank[i] * MATRIX_DIM);
    }

    std::vector<int> desplazamientos_A(nprocs);     // desplazamientos_A[4] = {0, 6 * 25, 12 * 25, 18 * 25};
    for (int i = 0; i < nprocs; i++){
        desplazamientos_A[i] = i * filas_por_rank[0] * MATRIX_DIM;
    }

    std::vector<int> desplazamientos_b(nprocs);     // desplazamientos_b[4] = {0, 6, 12, 18};
    for (int i = 0; i < nprocs; i++){
        desplazamientos_b[i] = i * filas_por_rank[0];
    }


    // Buffers
    std::unique_ptr<double[]> A; // Solo para RANK 0.
    std::unique_ptr<double[]> b; // Solo para RANK 0
    std::unique_ptr<double[]> x = std::make_unique<double[]>(MATRIX_DIM);

    // Buffers locales
    std::unique_ptr<double[]> A_local = std::make_unique<double[]>(filas_por_rank[rank] * MATRIX_DIM); // 7 x 25
    std::unique_ptr<double[]> b_local = std::make_unique<double[]>(filas_por_rank[rank]); // 7

    if (rank == 0){
        A = std::make_unique<double[]>(MATRIX_DIM * MATRIX_DIM);
        b = std::make_unique<double[]>(MATRIX_DIM);

        // Inicializar matriz A
        for (int i = 0; i < MATRIX_DIM; ++i)
        {
            printf("fila %d: ", i);
            for (int j = 0; j < MATRIX_DIM; ++j)
            {
                int index = i * MATRIX_DIM + j;
                A[index] = static_cast<double>(i);
                printf("%.0f ", A[index]);
            }
            printf("\n");
        }

        /* Inicializar el vector x. Solo el Rank 0 x = {1, 1, 1, ...}, en los demás procesos, como no se asigna
         * valores, x = {0, 0, 0, ...}
         */
        for (int i = 0; i < MATRIX_DIM; ++i)
        {
            x[i] = 1.0;
        }

        printf("\nDISPOSICION DE LOS DATOS:\n");
        printf("filas_por_rank: ");
        for (int i = 0; i < nprocs; i++){
            printf("%d ", filas_por_rank[i]);
        }
        printf("\n");

        printf("elementos_por_rank_A: ");
        for (int i = 0; i < nprocs; i++){
            printf("%d ", elementos_por_rank_A[i]);
        }
        printf("\n");

        printf("desplazamientos_A: ");
        for (int i = 0; i < nprocs; i++){
            printf("%d ", desplazamientos_A[i]);
        }
        printf("\n");

        printf("desplazamientos_b: ");
        for (int i = 0; i < nprocs; i++){
            printf("%d ", desplazamientos_b[i]);
        }
        printf("\n");
    }

    int rank_to_print = 0;
    // Imprimir vector x
    if (rank == rank_to_print)
    {
        printf("\nEl vector x es: \n");
        imprimir_vector(x.get(), MATRIX_DIM);
    }

    // Difundir el vector x a todos los procesos
    MPI_Bcast(x.get(), MATRIX_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(
        A.get(), elementos_por_rank_A.data(), desplazamientos_A.data(), MPI_DOUBLE, // Buffer de envío, tamaños y desplazamientos
        A_local.get(), elementos_por_rank_A[rank], MPI_DOUBLE, // Buffer de recepción y cantidad recibida
        0, MPI_COMM_WORLD);

    if (rank == rank_to_print)
    {
        printf("\nEn el Rank %d, la matriz A_local es:\n", rank);
        imprimir_matriz(A_local.get(), filas_por_rank[rank], MATRIX_DIM);
    }

    multiplicar_matriz(A_local.get(), x.get(), b_local.get(), filas_por_rank[rank], MATRIX_DIM);

    if (rank == rank_to_print)
    {
        printf("En el Rank %d, b_local es: \n", rank);
        imprimir_vector(b_local.get(), filas_por_rank[rank]);
    }

    MPI_Gatherv(b_local.get(), filas_por_rank[rank], MPI_DOUBLE,
                b.get(), filas_por_rank.data(), desplazamientos_b.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Imprimir el vector b, que es el resultado final.
    if (rank == 0)
    {
        printf("Rank 0, b: \n");
        imprimir_vector(b.get(), MATRIX_DIM);
    }

    MPI_Finalize();

    return 0;
}
