#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 2000      // Tamanho da grade global (linhas e colunas)
#define T 500       // Número de iterações no tempo
#define D 0.1       // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);                   // Inicializa o MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Obtém o identificador do processo
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Obtém o número total de processos

    // Verifica se o número de linhas é divisível pelo número de processos.
    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "O número de linhas N deve ser divisível pelo número de processos.\n");
        MPI_Finalize();
        return 1;
    }

    // Cada processo cuidará de um bloco de linhas.
    // Para facilitar as trocas de fronteira, alocamos duas linhas extra (linhas fantasma)
    int local_n = N / size;   // número de linhas *reais* do processo
    int local_rows = local_n + 2; // inclui linha de halo superior e inferior

    // Aloca as matrizes locais: C (concentração atual) e C_new (próxima iteração)
    double **C = malloc(local_rows * sizeof(double *));
    double **C_new = malloc(local_rows * sizeof(double *));
    if (C == NULL || C_new == NULL) {
        fprintf(stderr, "Erro de alocação de memória\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < local_rows; i++) {
        C[i] = malloc(N * sizeof(double));
        C_new[i] = malloc(N * sizeof(double));
        if (C[i] == NULL || C_new[i] == NULL) {
            fprintf(stderr, "Erro de alocação de memória\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Inicializa as matrizes com 0.0
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            C_new[i][j] = 0.0;
        }
    }

    // Define a condição inicial: concentração alta no centro da grade global.
    // A linha global do centro é global_center_row e a coluna é global_center_col.
    int global_center_row = N / 2;
    int global_center_col = N / 2;
    // Cada processo é responsável por um bloco de linhas consecutivas. O primeiro índice real deste processo é:
    int start_row = rank * local_n;
    int end_row = start_row + local_n - 1;
    if (global_center_row >= start_row && global_center_row <= end_row) {
        // O índice local correspondente é (global_center_row - start_row + 1)
        int local_i = global_center_row - start_row + 1; // +1 pois a linha 0 é halo
        C[local_i][global_center_col] = 1.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* ---------------------- Loop temporal ---------------------- */
    for (int t = 0; t < T; t++) {
        MPI_Status status;
        /* Troca de fronteiras (linhas halo):
           - Se não é o primeiro processo, envia a primeira linha real (linha 1) para o processo acima
             e recebe a linha de halo superior (linha 0) do vizinho.
           - Se não é o último processo, envia a última linha real (linha local_n) para o processo abaixo
             e recebe a linha de halo inferior (linha local_n+1) do vizinho.
        */
        if (rank > 0) {
            MPI_Sendrecv(C[1],         // envia a primeira linha real
                         N, MPI_DOUBLE,
                         rank - 1, 0,
                         C[0],         // recebe na linha de halo superior
                         N, MPI_DOUBLE,
                         rank - 1, 1,
                         MPI_COMM_WORLD,
                         &status);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(C[local_n],   // envia a última linha real
                         N, MPI_DOUBLE,
                         rank + 1, 1,
                         C[local_n + 1], // recebe na linha de halo inferior
                         N, MPI_DOUBLE,
                         rank + 1, 0,
                         MPI_COMM_WORLD,
                         &status);
        }

        // Calcula a atualização usando o método das diferenças finitas.
        // As células a atualizar são as linhas reais (i = 1 ... local_n) e colunas 1 ... N-2
        double local_diff = 0.0;
        for (int i = 1; i <= local_n; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4.0 * C[i][j]) 
                    / (DELTA_X * DELTA_X)
                );
                local_diff += fabs(C_new[i][j] - C[i][j]);
            }
        }
        // Copia os novos valores para C (exceto as bordas globais que permanecem 0)
        for (int i = 1; i <= local_n; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }

        // A cada 100 iterações, calcula a diferença média global.
        if (t % 100 == 0) {
            double global_diff;
            MPI_Reduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                double avg_diff = global_diff / ((N - 2) * (N - 2));
                printf("Interação %d - diferença média = %g\n", t, avg_diff);
            }
        }
    }
    /* ---------------------- Fim do loop temporal ---------------------- */

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // Recupera a concentração final no centro da grade.
    // Apenas o processo que contém a linha global 'global_center_row' envia esse valor para o processo 0.
    double center_val = 0.0;
    if (global_center_row >= start_row && global_center_row <= end_row) {
        int local_i = global_center_row - start_row + 1;
        center_val = C[local_i][global_center_col];
    }
    double final_center;
    MPI_Reduce(&center_val, &final_center, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Concentração final no centro: %f\n", final_center);
        printf("Tempo de execução (MPI): %f segundos\n", elapsed);
    }

    // Libera a memória alocada
    for (int i = 0; i < local_rows; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}
