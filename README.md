
# Simulação e Análise de Modelos de Difusão de Contaminantes

Este repositório contém os arquivos relacionados ao trabalho **"Relatório sobre a Simulação da Equação de Difusão utilizando OpenMP"**, incluindo a implementação da equação de difusão em sua versão sequencial e paralelizada.

## Descrição

O objetivo deste projeto é simular a equação de difusão utilizando o método das diferenças finitas para descrever o espalhamento de contaminantes em corpos d'água. A implementação foi feita em C, com uma versão sequencial e outra paralelizada utilizando a biblioteca **OpenMP**.

## Estrutura do Repositório

- **`relatorio.pdf`**: Relatório completo com explicações teóricas, implementação, análise de desempenho e resultados obtidos.
- **`codigo_nao_paralelizado.c`**: Código C da simulação sequencial (não paralelizado).
- **`codigo_paralelizado.c`**: Código C da simulação paralelizada utilizando OpenMP.

## Métodos e Implementação

### Método de Diferenças Finitas
A equação de difusão transiente em duas dimensões é representada como:

```math
\frac{{\partial C}}{{\partial t}} = D \cdot \nabla^2C
```

Essa equação é discretizada em uma grade bidimensional com aproximação por diferenças finitas centrais, resultando na seguinte fórmula:

```math
C_{{i,j}}^{{t+1}} = C_{{i,j}}^t + D \cdot \Delta t \cdot \frac{{C_{{i+1,j}}^t + C_{{i-1,j}}^t + C_{{i,j+1}}^t + C_{{i,j-1}}^t - 4C_{{i,j}}^t}}{{(\Delta x)^2}}
```

### Configuração
- Tamanho da grade: \( N = 2000 \)
- Iterações no tempo: \( T = 500 \)
- Coeficiente de difusão: \( D = 0.1 \)
- Passo temporal: \( \Delta t = 0.01 \)
- Passo espacial: \( \Delta x = 1.0 \)

### Diferenças entre as Implementações
- **Sequencial**: Executa as atualizações célula por célula, iterando na grade de forma tradicional.
- **Paralelizada (OpenMP)**: Utiliza a biblioteca OpenMP para paralelizar os loops que atualizam as células da grade e calculam a diferença média, distribuindo a carga de trabalho entre os núcleos disponíveis na CPU.

## Compilação e Execução

### Pré-requisitos
- Um compilador C com suporte a OpenMP (como `gcc`).
- Sistema operacional compatível (Linux, macOS ou Windows).

### Compilação
Para compilar os arquivos, utilize os comandos abaixo:

1. **Código sequencial**:
   ```bash
   gcc codigo_nao_paralelizado.c -o sequencial -lm
   ```

2. **Código paralelizado**:
   ```bash
   gcc -fopenmp codigo_paralelizado.c -o paralelizado -lm
   ```

### Execução
Após a compilação, execute os programas:

1. **Sequencial**:
   ```bash
   ./sequencial
   ```

2. **Paralelizado**:
   ```bash
   ./paralelizado
   ```

### Resultados
Os resultados incluem a concentração final no centro da grade e o tempo médio de execução para ambas as versões. Esses dados são analisados no relatório.

## Observações
- A versão paralelizada utiliza diretivas como `#pragma omp parallel for collapse(2)` e `reduction` para otimizar loops e evitar condições de corrida.
- Certifique-se de que seu sistema suporta o número de threads utilizado no OpenMP para obter resultados otimizados.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir _issues_ ou enviar _pull requests_.

## Licença
Este projeto está licenciado sob a [MIT License](LICENSE).

---

Desenvolvido por Amanda, Éric F. C. Yoshida, e Henrique C. Garcia - Instituto de Ciência e Tecnologia (UNIFESP).
