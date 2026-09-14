# Inspeção de Qualidade Industrial: Visão Computacional e CNNs

> Mini-Projeto Avaliativo — Módulo 2 | Curso SC TEC

Este projeto desenvolve um pipeline em Jupyter Notebook para automação da inspeção visual de peças. A solução combina **técnicas clássicas de Visão Computacional (OpenCV)** para análise exploratória e destacamento de defeitos com uma **Rede Neural Convolucional (CNN em TensorFlow/Keras)** para classificação automatizada em larga escala entre peças **OK** (`ok_front`) e **Defeituosas** (`def_front`).

---

## Estrutura do Repositório

```text
.
├── dataset/
│   └── casting_512x512.zip                 # Base de dados compactada
├── notebook/
│   └── SC_TEC_Mod2_Mini_Proj_CNN_OpenCV_Pecas.ipynb  # Script principal
├── LICENSE
└── README.md                              # Documentação do projeto

```

---

## Tecnologias e Bibliotecas Utilizadas

* **Linguagem & Ambiente:** Python 3.x, Jupyter Notebook / VS Code / Google Colab
* **Visão Computacional Clássica:** `OpenCV` (`cv2`)
* **Deep Learning:** `TensorFlow` / `Keras`
* **Manipulação de Dados e Visualização:** `NumPy`, `Matplotlib`, `Seaborn`
* **Métricas e Avaliação:** `Scikit-Learn`
* **Controles Interativos:** `ipywidgets`

---

## Funcionalidades e Sprints de Desenvolvimento

O projeto foi dividido em etapas alinhadas às necessidades industriais:

### 1. Ingestão e Análise Exploratória (EDA)

* **Extração Automática:** Download e descompactação do dataset `casting_512x512`.


* **Balanceamento de Classes:** O conjunto possui **1.300 imagens**:


    * `def_front`: 781 imagens (60.1%)


    * `ok_front`: 519 imagens (39.9%)


### 2. Visão Computacional Clássica (OpenCV)

Aplicação de filtros em amostras para destacar imperfeições físicas (trincas e ranhuras) antes de alimentar a inteligência artificial:

* **Conversão de Cores:** BGR para RGB (exibição) e BGR para Escala de Cinza (redução de dimensão).


* **Filtro de Suavização:** `GaussianBlur` (kernel 5x5) para remover pequenos ruídos da textura do metal.


* **Detecção de Bordas:** Algoritmo `Canny` para evidenciar contornos e falhas internas.


* **Limiarização (Thresholding):** Segmentação binária invertida (`THRESH_BINARY_INV`) para isolar a peça.


* **Operações Morfológicas:** Aplicação de Fechamento (`Closing`), Erosão (`Erode`) e Dilatação (`Dilate`) para unificação de contornos e remoção de falhas isoladas.


* **Controles Interativos:** Ajuste dinâmico de limiares via sliders do `ipywidgets`.



### 3. Pipeline de Dados e Data Augmentation (Keras)

* **Divisão Proporcional do Dataset:**
    * **70% Treino**

    * **15% Validação**

    * **15% Teste Cego** (dados nunca vistos pelo modelo durante o treino)




* **Otimização de Performance:** Pipeline otimizado com `.cache()`, `.shuffle()` e `.prefetch(buffer_size=AUTOTUNE)`.


* **Data Augmentation Dinâmico:** Aplicação no conjunto de treino com rotações aleatórias, espelhamento (horizontal e vertical), translação, zoom e variações de brilho e contraste para dar imunidade a mudanças de posicionamento na esteira.



### 4. Arquitetura da Rede Neural Convolucional (CNN)

A rede foi estruturada da seguinte forma:

1. **Entrada:** Resolução original $512 \times 512$ em escala de cinza.


2. **Redimensionamento e Normalização:** Redução para $300 \times 300$ pixels e reescalamento de pixels para a escala $[0, 1]$ via `Rescaling(1./255)`.


3. **Extração de Características (3 Blocos Convolucionais):**
* Bloco 1: `Conv2D` (16 filtros, $3 \times 3$) + `BatchNormalization` + `MaxPooling2D`.


* Bloco 2: `Conv2D` (32 filtros, $3 \times 3$) + `BatchNormalization` + `MaxPooling2D`.


* Bloco 3: `Conv2D` (64 filtros, $3 \times 3$) + `BatchNormalization` + `MaxPooling2D`.


* Regularização L2 adicionada em todas as camadas convolucionais.




4. **Classificação Densa:**
* Achatamento via `Flatten`.


* Camada `Dense` com 200 neurônios.


* Camada de `Dropout(0.5)` para evitar overfitting.


* Saída `Dense(1)` com ativação `sigmoid` para decisão binária.




5. **Compilação:** Otimizador `Adam` e função de perda `binary_crossentropy`.

#### Resumo da Estrutura do Modelo (`model.summary()`)

A execução da inspeção da rede exibe a seguinte distribuição de camadas e parâmetros:

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 300, 300, 1)       0         
 conv2d (Conv2D)             (None, 300, 300, 16)      160       
 batch_normalization         (None, 300, 300, 16)      64        
 max_pooling2d               (None, 150, 150, 16)      0         
 conv2d_1 (Conv2D)           (None, 150, 150, 32)      4,640      
 batch_normalization_1       (None, 150, 150, 32)      128       
 max_pooling2d_1             (None, 75, 75, 32)        0         
 conv2d_2 (Conv2D)           (None, 75, 75, 64)        18,496     
 batch_normalization_2       (None, 75, 75, 64)        256       
 max_pooling2d_2             (None, 37, 37, 64)        0         
 flatten (Flatten)           (None, 87616)             0         
 dense (Dense)               (None, 200)               17,523,400  
 dropout (Dropout)           (None, 200)               0         
 dense_1 (Dense)             (None, 1)                 201       
=================================================================
Total params: 17,547,345 (66.94 MB)
Trainable params: 17,547,121 (66.94 MB)
Non-trainable params: 224 (896.00 Byte)
_________________________________________________________________

```

---

### 5. Treinamento do Modelo

O treinamento foi configurado para garantir rápida convergência e proteção contra overfitting:

* **Épocas Definidas:** Até 40 épocas.
* **Estratégia de Interrupção Antecipada (`EarlyStopping`):**
    * Monitoramento da perda de validação (`val_loss`).
    * Paciência (`patience`) de 10 épocas sem melhora para parar o treino.
    * Configuração `restore_best_weights=True` para restaurar automaticamente os melhores pesos atingidos durante a execução.

* **Acompanhamento por Época:** Monitoramento simultâneo das métricas de acurácia (`accuracy`, `val_accuracy`) e perda (`loss`, `val_loss`).

---

## Resultados e Auditoria do Modelo

### Desempenho no Teste Cego

A avaliação foi realizada em um conjunto de **198 imagens** totalmente inéditas (nunca vistas pelo modelo durante o treino ou validação).

* **Acurácia Geral:** **~85,35%** (169 acertos de 198 amostras).
* **Total de Erros:** 29 imagens classificadas incorretamente.

---

### Curvas de Aprendizado (Acurácia e Perda)

A evolução do treinamento ao longo de 40 épocas demonstra um processo claro de ajuste e estabilização do modelo:

1. **Acurácia do Modelo:**
    * Nas primeiras 18 épocas, o modelo manteve um desempenho constante (treino estabilizado em ~61% e validação em ~44%).
    * A partir da época 19, ocorreu um avanço expressivo: a acurácia de validação subiu rapidamente, alcançando picos entre **80% e 85%**, frequentemente superando a acurácia de treino. Esse comportamento reflete a eficácia das transformações de *Data Augmentation* aplicadas no treino, tornando a validação mais consistente.


2. **Perda (Loss) do Modelo (Escala Logarítmica):**
    * O erro de treino caiu de forma acentuada logo na primeira época, mantendo-se em patamares baixos (abaixo de 1.0) e estáveis.
    * A perda de validação apresentou uma elevação inicial e um pico isolado perto da época 20, mas reduziu e convergiu rapidamente para a faixa de **0.5 a 0.6**, alinhando-se à perda de treino e confirmando que o modelo não sofreu overfitting severo.

---

### Matriz de Confusão e Relatório de Classificação

#### Matriz de Confusão (Dados Brutos)

|  | Predito: Defeituoso (`def_front`) | Predito: OK (`ok_front`) | Total Real |
| --- | --- | --- | --- |
| **Real: Defeituoso (`def_front`)** | **82** *(Verdadeiros Positivos)* | **27** *(Falsos Negativos)* | 109 |
| **Real: OK (`ok_front`)** | **2** *(Falsos Positivos)* | **87** *(Verdadeiros Negativos)* | 89 |

---

#### Relatório Detalhado de Métricas (`classification_report`)

| Classe | Precisão (Precision) | Revocação (Recall) | F1-Score | Suporte (Imagens) |
| --- | --- | --- | --- | --- |
| **Defeituoso (`def_front`)** | **0,98** (98%) | **0,75** (75%) | **0,85** (85%) | 109 |
| **OK (`ok_front`)** | **0,76** (76%) | **0,98** (98%) | **0,86** (86%) | 89 |
| **Média Ponderada (Weighted Avg)** | **0,88** (88%) | **0,85** (85%) | **0,85** (85%) | 198 |

---

### Análise Industrial dos Resultados

1. **Alta Precisão na Detecção de Defeitos (98%):** Quando o modelo classifica uma peça como defeituosa (`def_front`), ele está correto em **98% das vezes** (apenas 2 peças boas foram classificadas incorretamente como defeituosas).
2. **Alta Revocação nas Peças Conformes (98%):** O modelo identificou corretamente **98% de todas as peças em estado adequado** (`ok_front`), garantindo que peças boas quase nunca sejam descarcartadas por engano.
3. **Ponto de Atenção Industrial:** 27 peças com defeito acabaram sendo classificadas como OK (falsos negativos). Em linhas de produção críticas, isso indica uma oportunidade futura de reajustar o limiar de decisão (*threshold*) da sigmoid para favorecer o recall da classe defeituosa.
4. **Auditoria Visual de Erros:** O notebook conta com um módulo de inspeção que isola os 29 casos de divergência e exibe a imagem original juntamente com a confiança da rede, permitindo identificar o impacto de sombras e bordas tênues.
5. **Gráficos de Treinamento:** Curvas de acurácia e perda (loss em escala logarítmica) comparando o desempenho de Treino vs. Validação.

---

## Como Executar o Projeto

### Pré-requisitos

Certifique-se de ter o Python 3.9+ e o suporte a Jupyter Notebooks instalado:

```bash
pip install jupyter notebook tensorflow opencv-python matplotlib seaborn scikit-learn ipywidgets

```

### Passo a Passo

1. **Clonar o Repositório:**
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio

```


2. **Abrir e Executar o Notebook:**
* **No VS Code:** Abra a pasta do repositório no VS Code, navegue até `notebook/SC_TEC_Mod2_Mini_Proj_CNN_OpenCV_Pecas.ipynb` e execute as células em sequência ("Run All").
* **Via Jupyter Notebook / Lab:**
```bash
jupyter notebook notebook/SC_TEC_Mod2_Mini_Proj_CNN_OpenCV_Pecas.ipynb

```


* **No Google Colab:** Faça o upload do arquivo `SC_TEC_Mod2_Mini_Proj_CNN_OpenCV_Pecas.ipynb` para o Google Colab e execute as células. A célula inicial fará o download automático e descompactação do dataset via URL.

---

## Autor

**Daniel Heitor Mortari**

[GitHub](https://github.com/dnkamo)

Desenvolvido como atividade prática avaliativa do **Módulo 2 (Visão Computacional e Aprendizado Profundo)** do curso **SC TEC**.