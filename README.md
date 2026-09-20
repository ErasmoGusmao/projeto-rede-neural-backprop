# Rede neural com retropropagação manual

Projeto final da disciplina Matemática para Ciência de Dados (Especialização em Deep Learning,
CIn/UFPE). Implementa uma rede neural 2 → 4 → 1 com propagação direta e retropropagação escritas
à mão, operação por operação, a partir do `exemplo4.py` da Aula 04 (21/08/2026), e valida os
gradientes por diferenças finitas e contra o `autograd` do PyTorch. Entrega em 26/09/2026.

## Decisões de projeto

| Aspecto | Baseline (`exemplo4.py`) | Proposto |
|---|---|---|
| Base de dados | `make_moons(100)` | `make_circles(400, noise=0.1, factor=0.5)` |
| Arquitetura | 2 → 2 → 1 | 2 → 4 → 1 |
| Ativação oculta | sigmoide | tanh |
| Ativação de saída | sigmoide | sigmoide |
| Inicialização | U[0, 1] | Xavier/Glorot, semente fixa |
| Gradiente em lote | soma | média |
| Divisão treino/teste | ausente | 70/30 estratificada, `StandardScaler` ajustado no treino |
| Comparação | Keras | PyTorch (`autograd`) |

## Estrutura

```
projeto_rede_neural.ipynb                 notebook de entrega
notebooks/exemplos_aula/
  exemplo4.py                             código de referência da Aula 04 (baseline)
  exemplo01.ipynb ... exemplo03.ipynb     exemplos 1 a 3 da aula, reproduzidos e comentados
  exemplo02_convergencia/                 curvas do exemplo 2 para vários números de iterações
figuras/
  grafos/                                 grafos computacionais do baseline e da rede proposta
  exemplos/                               grafos computacionais dos exemplos 1 a 3
  fig_dataset.svg                         base de dados make_circles
scripts/                                  geradores dos grafos (networkx + matplotlib)
resultados/                               tabela consolidada dos experimentos da seção 10
requirements.txt                          ambiente Python
```

Os grafos de `figuras/` são gerados pelos scripts de `scripts/`; cada script grava PNG e SVG na
pasta correspondente. O notebook referencia as figuras por caminho relativo à raiz do projeto.

## Ambiente

Python 3.13. Criar o ambiente e instalar as dependências com:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O `requirements.txt` fixa numpy, scikit-learn, matplotlib, networkx, pandas e PyTorch (CPU).
Os grafos não dependem do Graphviz.

## Execução

Abrir `projeto_rede_neural.ipynb` e executar todas as células em ordem. A seção 10
(varredura de arquiteturas, ativações e taxas de aprendizado em PyTorch) leva alguns minutos em CPU.
