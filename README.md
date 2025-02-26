# Relatório: Implementação de uma CNN para Identificação de Vogais na Língua Brasileira de Sinais (LIBRAS)

**Universidade Federal do Ceará**  
**Disciplina: Inteligência Artificial**  
**Professor:** João Paulo do Vale Madeiro  
**Alunos:** Francisco Jean (541790) ^ Diego Caracas (542564) ^ João Gustavo (538609) ^ Levy Oliveira (541800)    

## Objetivo
O objetivo deste projeto é implementar uma **rede neural convolucional (CNN)** para a identificação das vogais na **Língua Brasileira de Sinais (Libras)**. Utilizamos a biblioteca **Keras** para criar e treinar o modelo, aproveitando camadas convolucionais para extrair padrões relevantes e camadas densas para realizar a classificação da melhor forma possível.

Exemplos de cada vogal presente no presente no dataset:

![image](https://github.com/user-attachments/assets/f5b992b8-4cb6-44f9-af1e-6b3db08bc46f)

## **Dataset: Libras Alphabet (Vogais)**  
**Fonte**: [Kaggle - Libras Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)  

### **Visão Geral do Dataset**  
O dataset contém imagens de gestos em **Libras** correspondentes às letras do alfabeto. Para o nosso propósito, focamos **apenas nas vogais**.  

#### **Estrutura do Dataset**  
- **Total de Classes**: 5 (uma para cada vogal).  
- **Imagens por Classe**: 3.000 imagens por vogal (total de 15.000 imagens).  
- **Formato das Imagens**: 200x200 pixels, RGB (3 canais de cor).  
- **Divisão**: É feito um split no dataset para se usar 90% das imagens para treino e 10% para teste.

### **Desafios**  
1. **Similaridade entre Classes**:  
   - Exemplo: O gesto para **"E"** (mão semi-fechada) e **"A"** (mão fechada) podem ser confundidos em certos ângulos.  
2. **Variações de Iluminação e Fundo**:  
   - Imagens foram capturadas em ambientes não controlados.  
3. **Diversidade de Tons de Pele e Tamanhos de Mão**:  
   - O dataset inclui participantes com diferentes características físicas.  

### **Mais Exemplos das Classes Existentes no Dataset**  
Abaixo estão exemplos ilustrativos dos gestos correspondentes a cada vogal:

#### **Classe "A"**  
- **Gesto**: Mão fechada em punho (polegar para o lado).  
- **Exemplo**:  
  ![A](https://raw.githubusercontent.com/Francisco-Jean/IA-TrabalhoFinal/refs/heads/FinalVersion/dataset/A/A1185.jpg) 

#### **Classe "E"**  
- **Gesto**: Mão semi-fechada, dedos curvados para dentro.  
- **Exemplo**:  
  ![E](https://github.com/Francisco-Jean/IA-TrabalhoFinal/blob/FinalVersion/dataset/E/E1010.jpg?raw=true)  

#### **Classe "I"**  
- **Gesto**: Mão aberta com dedo mindinho estendido e outros dedos dobrados.  
- **Exemplo**:  
  ![I](https://github.com/Francisco-Jean/IA-TrabalhoFinal/blob/FinalVersion/dataset/I/I1177.jpg?raw=true)

#### **Classe "O"**  
- **Gesto**: Mão formando um círculo (dedos curvados tocando o polegar).  
- **Exemplo**:  
  ![O](https://github.com/Francisco-Jean/IA-TrabalhoFinal/blob/FinalVersion/dataset/O/O113.jpg?raw=true) 

#### **Classe "U"**  
- **Gesto**: Dois dedos estendidos (indicador e médio) apontando para cima.  
- **Exemplo**:  
  ![U](https://github.com/Francisco-Jean/IA-TrabalhoFinal/blob/FinalVersion/dataset/U/U160.jpg?raw=true)  


### **Pré-processamento e Dificuldades**  
#### **Passos Críticos**  
1. **Redimensionamento**: Padronizar imagens para 128x128 pixels (equivalente ao `input_shape` do modelo).  
2. **Normalização**: Valores de pixel escalonados para [0, 1].  
3. **Data Augmentation**:  
   - Rotação (±20°), deslocamento horizontal/vertical (±10%) para generalização.  

#### **Dificuldades Específicas**  
- **Ângulos Não Padrão**: Gestos capturados de perspectivas laterais podem gerar ambiguidade.  
- **Sobreposição de Dedos**: Em gestos como "E" ou "O", dedos curvados podem se fundir visualmente.  

## O que é uma CNN?
As **Redes Neurais Convolucionais (CNNs)** são um tipo de rede neural especialmente eficaz para o processamento de imagens. Elas utilizam camadas convolucionais para extrair características automaticamente, reduzindo a necessidade de extração manual de features. As principais camadas de uma CNN são:

- **Camadas Convolucionais**: Aplicam filtros para detectar padrões como bordas, texturas e formas.
- **Função de Ativação**: Introduzir não-linearidade, permitindo que a CNN aprenda padrões complexos.
- **Camadas de Pooling**: Reduzem a dimensionalidade dos dados, tornando a rede mais eficiente e menos suscetível ao overfitting, preservando as informações mais importantes.
- **Camada Flatten**: Converte o mapa de características 2D em um vetor 1D para ser processado pelas camadas densas.
- **Camadas Totalmente Conectadas(Camadas Densas)**: Realizar a classificação final com base nas características extraídas.
- **Dropout**: Regularização que desativa aleatoriamente neurônios para evitar overfitting.

## CNN x Outros modelos
Dentre os benefício de utilziar da arquitetura CNN para problemas de processamento de imagens e aprendizado de máquina, podemos citar:
- Aproveita a estrutura espacial das imagens: 
  A CNN usa convoluções para identificar padrões locais como bordas, texturas e formas.

- Reduz a quantidade de parâmetros, prevenindo overfitting:
  CNNs compartilham filtros (kernels), reduzindo drasticamente o número de pesos.

- Reconhece objetos mesmo que sejam deslocados ou redimensionados:
  As camadas convolucionais e pooling tornam a CNN resistente a deslocamentos, rotações e redimensionamentos.
   
- É otimizada para computação paralela, permitindo treinar redes profundas com milhões de imagens:
  CNNs usam menos neurônios devido ao compartilhamento de pesos e pooling, tornando o treinamento mais rápido.
  Além disso, operações de convolução são altamente otimizadas para GPUs, permitindo treinamento eficiente.

## Nossa CNN x Arquitetura VGG

| Critério                    | Sua CNN                         | VGG                                    |
|-----------------------------|---------------------------------|----------------------------------------|
| **Número de parâmetros**    | Menos parâmetros (mais leve)    | Muitos parâmetros (mais pesado)        |
| **Profundidade**            | 3 camadas convolucionais        | 5 blocos convolucionais                |
| **Aprendizado de features** | Bom, mas menos detalhado        | Melhor aprendizado de características  |
| **Overfitting**             | Menor risco                     | Maior risco devido a muitos parâmetros |
| **Velocidade de treino**    | Mais rápido                     | Mais lento                             |
| **Necessidade de dados**    | Menos dados necessários         | Precisa de mais dados                  |
| **Precisão final**          | Boa para problemas simples      | Melhor para problemas complexos        |


## Modelo Implementado
A implementação da CNN foi feita utilizando **Keras** e segue uma arquitetura com as seguintes camadas:

### 1. Camadas Convolucionais e de Pooling
1. **Primeira camada convolucional:**
   - Convolução com **64 filtros**, tamanho de kernel **3x3**, stride **1** e ativação **ReLU**.
   - Essa camada aprende a detectar padrões básicos como bordas e texturas.
   - Input shape definido pela variável `target_dims`, que representa as dimensões da imagem de entrada, no nosso caso `128x128`
   
2. **Camada de Pooling:**
   - **MaxPooling2D** com tamanho de janela **2x2**.
   - Reduz a dimensionalidade ao manter apenas os valores máximos dentro da janela, preservando as características mais importantes e reduzindo o risco de overfitting.
   
3. **Segunda camada convolucional:**
   - Convolução com **128 filtros**, kernel **3x3**, stride **1** e ativação **ReLU**.
   - Detecta características mais complexas combinando as informações extraídas da camada anterior.
   
4. **Camada de Pooling:**
   - **MaxPooling2D** com tamanho de janela **2x2**.
   - Mantém apenas as informações mais relevantes para a classificação final.
   
5. **Terceira camada convolucional:**
   - Convolução com **256 filtros**, kernel **3x3**, stride **1** e ativação **ReLU**.
   - Essa camada é responsável por capturar padrões avançados, como formas mais específicas dos gestos representando as vogais.
   
6. **Camada de Pooling:**
   - **MaxPooling2D** com tamanho de janela **2x2**.
   - Reduz a complexidade do modelo, garantindo um melhor desempenho no treinamento e inferência.

### 2. Camadas Densas
7. **Flatten:**
   - Converte a saída das camadas convolucionais em um vetor unidimensional, preparando os dados para as camadas densas.

8. **Camada totalmente conectada:**
   - **256 neurônios** com ativação **ReLU**.
   - Realiza uma combinação não-linear das características extraídas para tomada de decisão.

9. **Dropout (0.5)**
   - Durante o treinamento, **50% dos neurônios** são desativados aleatoriamente.
   - Isso ajuda a reduzir overfitting e melhora a generalização do modelo para novos dados.

10. **Camada de Saída:**
   - Contém **`num_classes` neurônios** com ativação **Softmax**.
   - Produz a probabilidade de cada classe (uma para cada vogal em Libras), permitindo a classificação correta do gesto.

### Compilação do Modelo
O modelo é compilado com os seguintes parâmetros:
- **Otimizador:** Adam (ajusta os pesos da rede de forma eficiente para minimizar o erro)
- **Função de perda:** Categorical Crossentropy (mede a diferença entre as previsões do modelo e os valores reais)
- **Métrica de avaliação:** Acurácia.

## Métricas Utilizadas
Para avaliar o desempenho do modelo, utilizamos as seguintes métricas:

-  **Loss (Perda):** Mede a diferença entre as previsões do modelo e os valores reais (usando `categorical_crossentropy`).
- **Acurácia:** A acurácia é a proporção de previsões corretas feitas pelo modelo em relação ao total de exemplos. Ela é útil para medir o desempenho geral do modelo em tarefas de classificação (`acuracia = (verdadeirosPosiitivos + falsosPositivos)/totalexemplos`).
  
- **Gráficos de Acurácia e Perda:** Demonstram o progresso do treinamento e possíveis detecções de overfitting.
![graficos_acc_loss](https://github.com/user-attachments/assets/568ea9c6-6c05-4545-a07d-51b2811873b9)

- **Tabela De Métricas (sklearn.metrics)** <br>
![classification](https://github.com/user-attachments/assets/5dea6a6d-eebf-4202-bfea-7b9c1ca9f5f1)

Essas métricas são amplamente utilizadas para avaliar o desempenho de modelos de classificação, especialmente em situações de classes desbalanceadas.

#### 1. **Recall** (Sensibilidade ou Taxa de Verdadeiros Positivos)
O **Recall** é a proporção de instâncias relevantes (positivas) que foram corretamente identificadas pelo modelo. Ou seja, ele mede a capacidade do modelo de **não deixar passar** exemplos positivos. 

**Fórmula**:

`recall = verdadeirosPositivos/(verdadeirosPositivos + falsosNegativos)`

- **Verdadeiros Positivos (TP)**: Exemplos corretamente classificados como positivos.
- **Falsos Negativos (FN)**: Exemplos positivos incorretamente classificados como negativos.

#### 2. **Precision** (Precisão)
A **Precision** é a proporção de instâncias que o modelo classificou como positivas e que realmente são positivas. Em outras palavras, ela mede a **precisão** quando o modelo diz que algo é positivo.

**Fórmula**:

`precision = verdadeirosPositivos/(verdadeirosPositivos + falsosPositivos)`

- **Verdadeiros Positivos (TP)**: Exemplos corretamente classificados como positivos.
- **Falsos Positivos (FP)**: Exemplos negativos incorretamente classificados como positivos.

#### 3. **F1-Score**
O **F1-Score** é a média harmônica entre Precision e Recall. Ele fornece uma métrica balanceada que leva em consideração tanto a **precisão** quanto a **sensibilidade** do modelo, sendo útil especialmente quando as classes estão desbalanceadas. Quanto mais próximo de 1, melhor é o desempenho do modelo.

**Fórmula**:

`f1-score = (2 x precision x recall)\(precision + recall)`


- O **F1-Score** busca um equilíbrio entre Precision e Recall, penalizando modelos que tenham um bom desempenho em apenas uma das métricas.

**Obs: As figuras acima apresentam os resultados para um teste com `batchsize = 32` e `epochs = 10`**. No relatório de classificação, os números 0, 1,..., 4 representam as classes A, E,..., U.

## Sobre a Implementação 

### Como Usar o Modelo Treinado?
Por conta de restrições no tamanho dos dados possíveis de armazenamento no GitHub, precisamos dividir o nosso modelo treinado em algumas partes, sendo necessário assim para efetuar a execução dele, executar o seguinte comando:

   ```bash
   cat model_part_* > libravogaisneuralnet.weights.h5
   ```

#### Obs: O modelo foi treinado no WSL.

## Conclusão
Este modelo CNN foi projetado para a identificação das vogais na **Língua Brasileira de Sinais**, extraindo características relevantes através de camadas convolucionais profundas. O uso de **dropout** e **otimizador Adam** ajuda na regularização e eficiência do treinamento. O desempenho final pode ser melhorado ajustando parâmetros importantes e analisando de maneira mais detalhada o dataset.  


