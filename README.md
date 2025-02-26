# Relatório: Implementação de uma CNN para Identificação de Vogais na Língua Brasileira de Sinais

**Universidade Federal do Ceará**  
**Disciplina: Inteligência Artificial**  
**Professor:** [Nome do Professor]  
**Alunos:** [Nomes dos Alunos]  
**Data:** [Data de Entrega]  

## Como Usar o Modelo Treinado?
   ```bash
   cat model_part_* > libravogaisneuralnet.weights.h5
   ```

## Objetivo
O objetivo deste projeto é implementar uma **rede neural convolucional (CNN)** para a identificação das vogais na **Língua Brasileira de Sinais (Libras)**. Utilizamos a biblioteca **Keras** para criar e treinar o modelo, aproveitando camadas convolucionais para extrair padrões visuais relevantes e camadas densas para realizar a classificação final.

## O que é uma CNN?
As **Redes Neurais Convolucionais (CNNs)** são um tipo de rede neural especialmente eficaz para o processamento de imagens. Elas utilizam camadas convolucionais para extrair características automaticamente, reduzindo a necessidade de extração manual de features. As principais camadas de uma CNN são:

- **Camadas Convolucionais**: Aplicam filtros para detectar padrões como bordas, texturas e formas.
- **Camadas de Pooling**: Reduzem a dimensionalidade dos dados, tornando a rede mais eficiente e menos suscetível ao overfitting.
- **Camadas Densas**: Responsáveis por tomar decisões baseadas nas features extraídas.
- **Dropout**: Regularização que desativa aleatoriamente neurônios para evitar overfitting.

## Nosso Modelo
A implementação da CNN foi feita utilizando **Keras** e segue a seguinte arquitetura:

### Camadas Convolucionais
1. **Primeira camada convolucional:**
   - 64 filtros, kernel 4x4, stride 1, ativação ReLU
   - Input shape definido pela variável `target_dims`
   
2. **Segunda camada convolucional:**
   - 64 filtros, kernel 4x4, stride 2, ativação ReLU
   - Reduz a dimensão da imagem
   
3. **Dropout (0.5)** para evitar overfitting.

4. **Terceira camada convolucional:**
   - 128 filtros, kernel 4x4, stride 1, ativação ReLU
   
5. **Quarta camada convolucional:**
   - 128 filtros, kernel 4x4, stride 2, ativação ReLU
   - Reduz a dimensionalidade novamente
   
6. **Dropout (0.5)** para regularização.

7. **Quinta camada convolucional:**
   - 256 filtros, kernel 4x4, stride 1, ativação ReLU

8. **Sexta camada convolucional:**
   - 256 filtros, kernel 4x4, stride 2, ativação ReLU

### Camadas Densas
9. **Flatten:**
   - Converte a saída da camada convolucional em um vetor 1D para a camada densa

10. **Dropout (0.5)** para regularização.

11. **Camada totalmente conectada:**
   - 512 neurônios, ativação ReLU

12. **Camada de Saída:**
   - `num_classes` neurônios com ativação **Softmax** para classificação

### Compilação do Modelo
O modelo é compilado com o otimizador **Adam**, função de perda **categorical_crossentropy** e a métrica de avaliação **accuracy**.

## Métricas Utilizadas
Para avaliar o desempenho do modelo, utilizamos as seguintes métricas:

- **Loss (Perda):** Mede a diferença entre as previsões do modelo e os valores reais (usando `categorical_crossentropy`).
- **Acurácia:** Mede a proporção de previsões corretas em relação ao total de exemplos.
- **Gráficos de Acurácia e Perda:** Normalmente, são utilizados para visualizar o progresso do treinamento e detectar sinais de overfitting.

## Conclusão
Este modelo CNN foi projetado para a identificação das vogais na **Língua Brasileira de Sinais**, extraindo características relevantes através de camadas convolucionais profundas. O uso de **dropout** e **otimizador Adam** ajuda na regularização e eficiência do treinamento. O desempenho final pode ser melhorado ajustando hiperparâmetros ou experimentando técnicas adicionais como **batch normalization** ou **data augmentation**.

