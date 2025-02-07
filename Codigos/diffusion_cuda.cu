#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <chrono>


#define N 2000    // Tamanho da grade
#define T 500     // Número de iterações no tempo
#define D 0.1     // Coeficiente de difusão
#define DELTA_T 0.01 // Passo de tempo
#define DELTA_X 1.0 // Espaçamento entre células

// Kernel CUDA para calcular a próxima iteração
__global__ void diffusion_step(double *C, double *C_new, int n) { 
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // Ignora a borda
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // Ignora a borda

    if (i < n - 1 && j < n - 1) { // Equação de difusão ignorando a borda
        C_new[i * n + j] = C[i * n + j] + D * DELTA_T * (
            (C[(i + 1) * n + j] + C[(i - 1) * n + j] + C[i * n + (j + 1)] + C[i * n + (j - 1)] - 4 * C[i * n + j]) /
            (DELTA_X * DELTA_X)
        ); 
    }
}

// Kernel para atualizar a matriz C e calcular a diferença média
__global__ void update_matrix(double *C, double *C_new, double *dif, int n) {
    __shared__ double local_dif[256]; // 16x16

    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; 
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    double diff = 0.0;

    if (i < n - 1 && j < n - 1) { 
        diff = fabs(C_new[i * n + j] - C[i * n + j]); // Diferença absoluta
        C[i * n + j] = C_new[i * n + j]; // Atualiza a matriz
    }

    local_dif[idx] = diff; // Armazena a diferença local

    __syncthreads(); // Sincronização para garantir que todos os threads tenham calculado a diferença

    // Redução para calcular a soma total
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) { 
        if (idx < stride) { 
            local_dif[idx] += local_dif[idx + stride]; // Soma a diferença
        }
        __syncthreads(); // Sincronização para garantir que todos os threads tenham calculado a soma
    }

    if (idx == 0) { 
        atomicAdd(dif, local_dif[0]); // Atualiza a diferença total
    }
}

int main() {
    size_t size = N * N * sizeof(double); // Tamanho da matriz
    double *C, *C_new, *d_C, *d_C_new, *d_dif; // Matrizes no host e no dispositivo
    double dif = 0.0; // Diferença média

    // Alocar memória no host
    C = (double *)malloc(size); // Matriz de concentração
    C_new = (double *)malloc(size); // Matriz de concentração na próxima iteração

    // Inicializar matrizes no host
    for (int i = 0; i < N; i++) { // Inicializa a matriz com zeros
        for (int j = 0; j < N; j++) { 
            C[i * N + j] = 0.0;
            C_new[i * N + j] = 0.0;
        }
    }

    // Inicializar concentração alta no centro
    C[N / 2 * N + N / 2] = 1.0;

    // Alocar memória no dispositivo
    cudaMalloc(&d_C, size); // Matriz de concentração
    cudaMalloc(&d_C_new, size); // Matriz de concentração na próxima iteração
    cudaMalloc(&d_dif, sizeof(double)); // Diferença média

    // Copiar dados para o dispositivo
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice); // Copia a matriz de concentração
    cudaMemcpy(d_C_new, C_new, size, cudaMemcpyHostToDevice); // Copia a matriz de concentração na próxima iteração

    dim3 blockDim(16, 16); // Bloco 16x16
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // Grid

    auto start = std::chrono::high_resolution_clock::now(); // Inicia a contagem de tempo
    // Iterações no tempo
    for (int t = 0; t < T; t++) { 
        cudaMemset(d_dif, 0, sizeof(double)); // Zera a diferença média

        diffusion_step<<<gridDim, blockDim>>>(d_C, d_C_new, N); // Calcula a próxima iteração
        cudaDeviceSynchronize(); // Sincroniza a execução dos kernels

        update_matrix<<<gridDim, blockDim>>>(d_C, d_C_new, d_dif, N); // Atualiza a matriz e calcula a diferença média
        cudaDeviceSynchronize(); // Sincroniza a execução dos kernels

        cudaMemcpy(&dif, d_dif, sizeof(double), cudaMemcpyDeviceToHost); // Copia a diferença média de volta para o host

        if (t % 100 == 0) { // Exibe a diferença média a cada 100 iterações
            printf("Iteracao %d - Diferenca media = %g\n", t, dif / ((N - 2) * (N - 2)));
        }
    }

    // Copiar dados de volta para o host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); // Copia a matriz de concentração
    auto stop = std::chrono::high_resolution_clock::now(); // Finaliza a contagem de tempo
    std::chrono::duration<double> elapsed = stop - start; // Calcula o tempo de execução

    // Exibir concentração final no centro
    printf("Concentracao final no centro: %f\n", C[N / 2 * N + N / 2]); 
    printf("Tempo de execucao (CUDA): %f segundos\n", elapsed.count());

    // Liberar memória
    free(C);
    free(C_new);
    cudaFree(d_C);
    cudaFree(d_C_new);
    cudaFree(d_dif);

    return 0;
}
