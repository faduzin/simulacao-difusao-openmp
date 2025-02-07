#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 2000  // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1  // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new) { // Função para resolver a equação de difusão
    for (int t = 0; t < T; t++) { // Iterar no tempo
        for (int i = 1; i < N - 1; i++) { // Iterar na grade
            for (int j = 1; j < N - 1; j++) { // Iterar na grade
                C_new[i][j] = C[i][j] + D * DELTA_T * ( // Resolver a equação de difusão
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X) 
                );
            }
        }
        // Atualizar matriz para a próxima iteração
        double difmedio = 0.; // Diferença média entre a matriz atual e a matriz anterior
        for (int i = 1; i < N - 1; i++) { // Iterar na grade
            for (int j = 1; j < N - 1; j++) { // Iterar na grade
                difmedio += fabs(C_new[i][j] - C[i][j]); // Calcular a diferença média
                C[i][j] = C_new[i][j];
            }
        }
        if ((t%100) == 0) 
            printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2))); // Exibir a diferença média
    }
}

int main() { 
    // Concentração inicial
    double **C = (double **)malloc(N * sizeof(double *)); // Matriz de concentração
    if (C == NULL) { // Verificar se a alocação de memória foi bem sucedida
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < N; i++) { // Alocar memória para a matriz de concentração
        C[i] = (double *)malloc(N * sizeof(double)); 
        if (C[i] == NULL) { // Verificar se a alocação de memória foi bem sucedida
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    }
    for (int i = 0; i < N; i++) { // Inicializar a matriz de concentração
        for (int j = 0; j < N; j++) { 
            C[i][j] = 0.;
        }
    }

    // Concentração para a próxima iteração
    double **C_new = (double **)malloc(N * sizeof(double *)); // Matriz de concentração para a próxima iteração
    if (C_new == NULL) { // Verificar se a alocação de memória foi bem sucedida
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < N; i++) { // Alocar memória para a matriz de concentração para a próxima iteração
        C_new[i] = (double *)malloc(N * sizeof(double));
        if (C_new[i] == NULL) { // Verificar se a alocação de memória foi bem sucedida
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    }
    for (int i = 0; i < N; i++) { // Inicializar a matriz de concentração para a próxima iteração
        for (int j = 0; j < N; j++) { 
            C_new[i][j] = 0.;
        }
    }

    // Inicializar uma concentração alta no centro
    C[N/2][N/2] = 1.0; 

    // Executar as iterações no tempo para a equação de difusão
	clock_t start = clock(); // Iniciar a contagem do tempo
	struct timeval inicio, final2; // Variáveis para medir o tempo de execução
	int tmili; // Variável para medir o tempo de execução
	gettimeofday(&inicio, NULL); // Iniciar a contagem do tempo
    diff_eq(C, C_new); // Resolver a equação de difusão

	gettimeofday(&final2, NULL); // Finalizar a contagem do tempo
	tmili = (int) (1000 * (final2.tv_sec - inicio.tv_sec) + (final2.tv_usec - inicio.tv_usec) / 1000); // Calcular o tempo de execução
    
    // Exibir resultado para verificação
    printf("Concentração final no centro: %f\n", C[N/2][N/2]);
    printf("\nTempo de execucao (Serial): %f segundos\n", (double) tmili/1000);
    
    return 0;
}
