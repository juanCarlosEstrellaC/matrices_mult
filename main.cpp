#include <memory>
#include <iostream>
#include <cmath>
#include <mpi.h>

#define MATRIX_DIM 25

// Función para imprimir un vector
void imprimir_vector(const double* v, int size)
{
    std::cout << "{ ";
    for (int i = 0; i < size; ++i)
    {
        std::cout << v[i] << (i < size - 1 ? ", " : " ");
    }
    std::cout << "}" << std::endl;
}

void imprimir_matriz(double* A, int rows, int cols)
{
    for (int i = 0; i < rows; i++){
        printf("i:", i);
        for (int j = 0; j < cols; j++){
            int index = i * cols + j;
            printf("%f ", A[index]);
        }
        printf("\n");
    }
}

void multiplicar_matriz(double* A, double* x, double* b,  int rows, int cols){
    for (int i = 0; i < rows; i++){
        b[i] = 0;
        for (int j = 0; j < cols; j++){
            int index = i * cols + j;
            b[i] += A[index] * x[j];
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int rows_per_rank; // Número de filas procesadas por cada proceso.
    int rows_alloc = MATRIX_DIM; // Tamaño final de la matriz. Por defecto es igual a MATRIX_DIM.
    int padding = 0; // padding = relleno. Por defecto es 0.

    // Ajustar el tamaño de la matriz de ser necesario.
    if (MATRIX_DIM % nprocs != 0)
    {
        // Si el residuo es diferente de 0, ajustar. Ejemplo: 25 % 4 = 1, ajustar con ceil.
        rows_alloc = std::ceil((double)MATRIX_DIM / nprocs) * nprocs; // 25 / 4 = 6.25, ceil(6.25) = 7, 7 * 4 = 28
        padding = rows_alloc - MATRIX_DIM; // 28 - 25 = 3
    }
    rows_per_rank = rows_alloc / nprocs; // 28 / 4 = 7

    // Tomar en cuenta que la matriz A es en un vector 1D, ie, A[rows_alloc * MATRIX_DIM]
    // A*x = b
    // A(rows_alloc x MATRIX_DIM) * x(MATRIX_DIM x 1) = b(rows_alloc x 1)
    // A(28 x 25) * x(25 x 1) = b(28 x 1)

    // Buffers
    std::unique_ptr<double[]> A; // Solo para RANK 0.
    std::unique_ptr<double[]> b; // Solo para RANK 0
    std::unique_ptr<double[]> x = std::make_unique<double[]>(MATRIX_DIM);

    // Buffers locales
    std::unique_ptr<double[]> A_local;
    std::unique_ptr<double[]> b_local;

    if (rank == 0)
    {
        std::cout << "dimension: " << MATRIX_DIM
            << ", rows_alloc=" << rows_alloc
            << ", rows_per_rank=" << rows_per_rank
            << ", padding=" << padding << std::endl;

        A = std::make_unique<double[]>(rows_alloc * MATRIX_DIM);
        b = std::make_unique<double[]>(rows_alloc);

        // Inicializar matriz A y vector b
        for (int i = 0; i < MATRIX_DIM; ++i)
        {
            std::cout << "fila " << i << ": ";
            for (int j = 0; j < MATRIX_DIM; ++j)
            {
                int index = i * MATRIX_DIM + j;
                A[index] = static_cast<double>(i);
                std::cout << A[index] << " ";
            }
            std::cout << std::endl;
        }

        // Inicializar el vector x
        for (int i = 0; i < MATRIX_DIM; ++i)
        {
            x[i] = 1.0;
        }
    }

    // Inicializar matrices locales
    A_local = std::make_unique<double[]>(rows_per_rank * MATRIX_DIM); // 7 x 25
    b_local = std::make_unique<double[]>(rows_per_rank); // 7

    // Imprimir vector x
    if (rank == 3)
    {
        printf("El vector x es: \n");
        imprimir_vector(x.get(), MATRIX_DIM);
    }

    // Difundir el vector x a todos los procesos
    MPI_Bcast(x.get(), MATRIX_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // enviar A con Scatter. Se envía la matriz A a todos los procesos en bloques de 7 filas y 25 columnas cada uno.
    // Divide el arreglo de entrada (A) en bloques iguales (o casi iguales) para distribuirlos entre todos los procesos.
    MPI_Scatter(A.get(),
        rows_per_rank * MATRIX_DIM, MPI_DOUBLE, // Enviar 7 x 25
        A_local.get(),rows_per_rank * MATRIX_DIM, MPI_DOUBLE, // Recibir 7 x 25
        0,
        MPI_COMM_WORLD);

    int rank_to_print = 0;
    if (rank == rank_to_print)
    {
        printf("La matriz A es: \n");
        imprimir_matriz(A_local.get(), rows_per_rank, MATRIX_DIM);
    }


    // calcular b_local = A_local * x
    int rows_per_rank_tmp = rows_per_rank;
    if (rank == nprocs - 1){
        rows_per_rank_tmp -= padding;
    }

    multiplicar_matriz(A_local.get(), x.get(), b_local.get(), rows_per_rank_tmp, MATRIX_DIM);

    if (rank == rank_to_print)
    {
        printf("rank: %d, b_local: \n", rank);
        imprimir_vector(b_local.get(), rows_per_rank);
    }

    // Enviar los resultados parciales al Rank 0 con Gather
    MPI_Gather(b_local.get(), rows_per_rank, MPI_DOUBLE,
        b.get(), rows_per_rank, MPI_DOUBLE,
        0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        printf("Rank 0, b: \n");
        imprimir_vector(b.get(), MATRIX_DIM);
    }


    MPI_Finalize();

    return 0;
}
