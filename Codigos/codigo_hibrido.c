#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 2000 // Tamanho global da grade (linhas e colunas)
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão
#define DELTA_T 0.01 // Intervalo de tempo
#define DELTA_X 1.0 // Espaçamento entre células
#define NUM_THREADS 8 // Número de threads

// Macro para checagem de erros nas chamadas CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) { // Função para checar erros
    if (code != cudaSuccess) { // Se houver erro
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); // Imprime o erro
        if (abort) exit(code);
    }
}

/*
    Kernel CUDA para atualizar os valores internos da grade e computar a diferença
    em cada célula.
    
    A ideia é que cada thread atualize uma célula (i,j) da sub-região real (excluindo os halos):
      - i varia de 1 a local_n   (ou seja, usamos um índice interno "i" de 0 a local_n-1 e depois
        somamos 1 para obter a linha real dentro da área alocada, visto que a linha 0 é halo).
      - j varia de 1 a N-2 (pois as colunas 0 e N-1 são as fronteiras globais).
      
    O kernel escreve o novo valor na matriz d_C_new e armazena em d_diff o valor absoluto da
    diferença entre o novo valor e o antigo.
*/
__global__ void update_kernel(const double *d_C, double *d_C_new, double *d_diff,
                              int local_n, int N, double D, double DELTA_T, double DELTA_X) {
    // Para a atualização, cada thread é responsável por uma célula interior:
    // i: índice de linha real = threadIdx.y + blockIdx.y*blockDim.y + 1   (varia de 1 a local_n)
    // j: índice de coluna real = threadIdx.x + blockIdx.x*blockDim.x + 1   (varia de 1 a N-2)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // j "local" para célula (0 ... N-3) corresponde a coluna = j+1
    int i = blockIdx.y * blockDim.y + threadIdx.y; // i "local" para célula (0 ... local_n-1) corresponde a linha = i+1

    if(i < local_n && j < (N - 2)) {
        int row = i + 1; // linha real dentro da sub-região (contando com halo superior na linha 0)
        int col = j + 1; // coluna real (entre 1 e N-2)
        int idx = row * N + col; // índice global da célula (row, col) na grade global
        double old_val = d_C[idx]; // valor atual da concentração na célula (row, col)
        double new_val = old_val + D * DELTA_T * (
            (d_C[(row+1)*N + col] + d_C[(row-1)*N + col] +
             d_C[row*N + (col+1)]   + d_C[row*N + (col-1)] - 4.0 * old_val)
            / (DELTA_X * DELTA_X)
        ); // novo valor da concentração na célula (row, col)
        d_C_new[idx] = new_val; // Atualiza o valor na matriz d_C_new
        // Armazena a diferença para posterior redução. Usamos um vetor d_diff de tamanho (local_n x (N-2))
        d_diff[i * (N - 2) + j] = fabs(new_val - old_val); 
    }
}

int main(int argc, char **argv) {
    int rank, size; // Identificador do processo e número total de processos
    omp_set_num_threads(NUM_THREADS); // Define o número de threads

    MPI_Init(&argc, &argv); // Inicializa o ambiente MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Pega o rank do processo
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Pega o número total de processos

    // Para simplificar, exige que N seja divisível pelo número de processos.
    if (N % size != 0) { // Se N não for divisível pelo número de processos
        if (rank == 0) // Se for o processo 0
            fprintf(stderr, "Erro: N deve ser divisível pelo número de processos.\n"); // Imprime o erro
        MPI_Finalize();
        exit(1);
    }

    // Cada processo manipula um bloco de linhas reais da grade
    int local_n = N / size;       // número de linhas reais (sem contar halos)
    int local_rows = local_n + 2;   // inclui uma linha de halo acima e outra abaixo

    size_t grid_size = local_rows * N * sizeof(double);

    // Aloca na GPU duas matrizes para a concentração: d_C (atual) e d_C_new (próxima iteração)
    double *d_C, *d_C_new; // Ponteiros para as matrizes na GPU 
    cudaCheckError(cudaMalloc((void **)&d_C, grid_size)); // Aloca memória para a matriz d_C
    cudaCheckError(cudaMalloc((void **)&d_C_new, grid_size)); // Aloca memória para a matriz d_C_new

    // Aloca um buffer no host para inicializar a grade.
    double *h_C = (double *) malloc(grid_size); // Aloca memória para a matriz h_C no host
    if (h_C == NULL) { // Se houver erro na alocação
        fprintf(stderr, "Erro de alocação no host\n"); // Imprime o erro
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Inicializa com 0.0 usando OpenMP para acelerar
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < local_rows; i++) { // Para cada linha
        for (int j = 0; j < N; j++) { // Para cada coluna
            h_C[i * N + j] = 0.0;
        }
    }
    // Define condição inicial: concentração alta no centro da grade global
    int global_center_row = N / 2; // Linha do centro da grade global
    int global_center_col = N / 2; // Coluna do centro da grade global
    int start_row = rank * local_n; // índice da primeira linha real deste processo (grade global)
    int end_row = start_row + local_n - 1; // índice da última linha real deste processo (grade global)
    if (global_center_row >= start_row && global_center_row <= end_row) { // Se o centro estiver neste processo
        int local_i = global_center_row - start_row + 1; // índice local (considerando o halo superior)
        h_C[local_i * N + global_center_col] = 1.0; // Concentração alta no centro
    }
    // Copia a condição inicial para ambas as matrizes na GPU
    cudaCheckError(cudaMemcpy(d_C, h_C, grid_size, cudaMemcpyHostToDevice)); // Copia a matriz h_C para d_C
    cudaCheckError(cudaMemcpy(d_C_new, h_C, grid_size, cudaMemcpyHostToDevice)); // Copia a matriz h_C para d_C_new
    free(h_C);

    // Aloca memória na GPU para armazenar as diferenças de cada célula para redução.
    // Tamanho: local_n x (N-2) (pois só atualizamos as células interiores, excluindo as colunas de borda).
    size_t diff_size = local_n * (N - 2) * sizeof(double); // Tamanho do vetor de diferenças
    double *d_diff; // Ponteiro para o vetor de diferenças na GPU
    cudaCheckError(cudaMalloc((void **)&d_diff, diff_size)); // Aloca memória para o vetor de diferenças

    // Aloca um buffer no host para trazer os valores de diferença e fazer redução com OpenMP.
    double *h_diff = (double *) malloc(diff_size); // Aloca memória para o vetor de diferenças no host
    if (h_diff == NULL) { // Verifica se houve erro na alocação
        fprintf(stderr, "Erro de alocação no host para diferenças\n"); 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Define os parâmetros de lançamento do kernel: cada thread cuida de uma célula interior.
    dim3 blockDim(16, 16); // 16 threads por bloco
    dim3 gridDim((N - 2 + blockDim.x - 1) / blockDim.x, (local_n + blockDim.y - 1) / blockDim.y); // Grid 2D

    MPI_Barrier(MPI_COMM_WORLD); // Barreira para sincronizar os processos
    double start_time = MPI_Wtime(); // Inicia a contagem do tempo

    /* Loop temporal principal */
    for (int t = 0; t < T; t++) {
        MPI_Status status;
        /*
            Troca de halos entre processos: cada processo envia a sua primeira linha real e
            última linha real e recebe as linhas de halo dos vizinhos.
            Usa-se MPI_Sendrecv com ponteiros para memória da GPU (MPI CUDA‐aware).
            
            - Se não for o primeiro processo (rank > 0), envia a linha 1 (primeira linha real) e
              recebe a linha 0 (halo superior) do processo rank-1.
            - Se não for o último processo (rank < size-1), envia a linha local_n (última linha real)
              e recebe a linha local_n+1 (halo inferior) do processo rank+1.
        */
        if (rank > 0) { // Se não for o primeiro processo
            MPI_Sendrecv(d_C + (1 * N), N, MPI_DOUBLE, rank - 1, 0, // Envia a linha 1
                         d_C + (0 * N), N, MPI_DOUBLE, rank - 1, 1, // Recebe a linha 0
                         MPI_COMM_WORLD, &status); // Comunica com o processo rank-1
        }
        if (rank < size - 1) { // Se não for o último processo
            MPI_Sendrecv(d_C + (local_n * N), N, MPI_DOUBLE, rank + 1, 1, // Envia a linha local_n
                         d_C + ((local_n + 1) * N), N, MPI_DOUBLE, rank + 1, 0, // Recebe a linha local_n+1
                         MPI_COMM_WORLD, &status); // Comunica com o processo rank+1
        }

        // Lança o kernel CUDA para atualizar a região interior e computar as diferenças.
        update_kernel<<<gridDim, blockDim>>>(d_C, d_C_new, d_diff, local_n, N, D, DELTA_T, DELTA_X); // Chama o kernel
        cudaCheckError(cudaDeviceSynchronize()); // Espera a execução do kernel

        // A cada 100 iterações, traz os dados de diferença para o host e calcula a diferença média.
        if (t % 100 == 0) {
            cudaCheckError(cudaMemcpy(h_diff, d_diff, diff_size, cudaMemcpyDeviceToHost)); // Copia o vetor de diferenças para o host
            double local_diff = 0.0; // Variável para armazenar a diferença local
            int diff_count = local_n * (N - 2); // Número total de diferenças
            #pragma omp parallel for reduction(+:local_diff)
            for (int i = 0; i < diff_count; i++) { // Para cada diferença
                local_diff += h_diff[i]; // Soma a diferença local
            }
            double global_diff = 0.0; // Variável para armazenar a diferença global
            MPI_Reduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Reduz a diferença
            if (rank == 0) { // Se for o processo 0
                double avg_diff = global_diff / ((N - 2) * (N - 2)); // Calcula a diferença média
                printf("Iteração %d - diferença média = %g\n", t, avg_diff);
            }
        }

        // Troca os ponteiros para que a próxima iteração use a matriz atualizada.
        double *temp = d_C; // Troca os ponteiros
        d_C = d_C_new; // d_C_new passa a ser d_C
        d_C_new = temp; // d_C passa a ser d_C_new
    }
    /* Fim do loop temporal */

    MPI_Barrier(MPI_COMM_WORLD); // Barreira para sincronizar os processos
    double end_time = MPI_Wtime(); // Finaliza a contagem do tempo
    double elapsed = end_time - start_time; // Calcula o tempo de execução

    // Recupera o valor final da concentração no centro da grade.
    double center_val = 0.0;
    if (global_center_row >= start_row && global_center_row <= end_row) { // Se o centro estiver neste processo
        int local_i = global_center_row - start_row + 1;  // índice local (considerando o halo superior)
        cudaCheckError(cudaMemcpy(&center_val, // Copia o valor do centro
                                    d_C + (local_i * N + global_center_col), // Índice do centro
                                    sizeof(double),
                                    cudaMemcpyDeviceToHost)); // Copia o valor do centro para o host
    }
    double final_center = 0.0; // Variável para armazenar a concentração final no centro
    MPI_Reduce(&center_val, &final_center, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Reduz o valor do centro
    if (rank == 0) { // Exibe o resultado final
        printf("Concentração final no centro: %f\n", final_center);
        printf("Tempo de execução (MPI+OpenMP+CUDA): %f segundos\n", elapsed);
    }

    // Libera os recursos alocados na GPU e no host.
    cudaFree(d_C);
    cudaFree(d_C_new);
    cudaFree(d_diff);
    free(h_diff);

    MPI_Finalize();
    return 0;
}
