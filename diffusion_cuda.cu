#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <chrono>


#define N 2000    // Tamanho da grade
#define T 500     // Número de iterações no tempo
#define D 0.1     // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

// Kernel CUDA para calcular a próxima iteração
__global__ void diffusion_step(double *C, double *C_new, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < n - 1 && j < n - 1) {
        C_new[i * n + j] = C[i * n + j] + D * DELTA_T * (
            (C[(i + 1) * n + j] + C[(i - 1) * n + j] + C[i * n + (j + 1)] + C[i * n + (j - 1)] - 4 * C[i * n + j]) /
            (DELTA_X * DELTA_X)
        );
    }
}

// Kernel para atualizar a matriz C e calcular a diferença média
__global__ void update_matrix(double *C, double *C_new, double *dif, int n) {
    __shared__ double local_dif[256];

    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    double diff = 0.0;

    if (i < n - 1 && j < n - 1) {
        diff = fabs(C_new[i * n + j] - C[i * n + j]);
        C[i * n + j] = C_new[i * n + j];
    }

    local_dif[idx] = diff;

    __syncthreads();

    // Redução para calcular a soma total
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) {
        if (idx < stride) {
            local_dif[idx] += local_dif[idx + stride];
        }
        __syncthreads();
    }

    if (idx == 0) {
        atomicAdd(dif, local_dif[0]);
    }
}

int main() {
    size_t size = N * N * sizeof(double);
    double *C, *C_new, *d_C, *d_C_new, *d_dif;
    double dif = 0.0;

    // Alocar memória no host
    C = (double *)malloc(size);
    C_new = (double *)malloc(size);

    // Inicializar matrizes no host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            C_new[i * N + j] = 0.0;
        }
    }

    // Inicializar concentração alta no centro
    C[N / 2 * N + N / 2] = 1.0;

    // Alocar memória no dispositivo
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_C_new, size);
    cudaMalloc(&d_dif, sizeof(double));

    // Copiar dados para o dispositivo
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_new, C_new, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();
    // Iterações no tempo
    for (int t = 0; t < T; t++) {
        cudaMemset(d_dif, 0, sizeof(double));

        diffusion_step<<<gridDim, blockDim>>>(d_C, d_C_new, N);
        cudaDeviceSynchronize();

        update_matrix<<<gridDim, blockDim>>>(d_C, d_C_new, d_dif, N);
        cudaDeviceSynchronize();

        cudaMemcpy(&dif, d_dif, sizeof(double), cudaMemcpyDeviceToHost);

        if (t % 100 == 0) {
            printf("Iteracao %d - Diferenca media = %g\n", t, dif / ((N - 2) * (N - 2)));
        }
    }

    // Copiar dados de volta para o host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

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
