# -*- coding: utf-8 -*-
"""
Grafo computacional do exemplo 3 (neurônio / perceptron de 2 entradas).

Funções do exemplo03.ipynb:

    nn(w, x)         = sign(w0 + w1*x0 + w2*x1)
    grad_nn(w, x, d) -> [grad_w0, grad_w1, grad_w2]

Operações elementares do feedforward de `grad_nn`:

    s0 = w1 * x0
    s1 = w2 * x1
    s2 = s0 + s1
    y  = w0 + s2
    e  = y - d
    L  = e**2

E o backward, na ordem inversa:

    grad_L  = 1
    grad_e  = grad_L * e
    grad_y  = grad_e
    grad_s2 = grad_y
    grad_s0 = grad_s2         grad_s1 = grad_s2
    grad_w0 = grad_y
    grad_w1 = grad_s0 * x0    grad_w2 = grad_s1 * x1

Convenções (as mesmas dos grafos do notebook do projeto):
  - nó redondo       = variável de entrada ou parâmetro
  - nó retangular    = operação elementar
  - rótulo preto     = nome da variável do código (fluxo direto)
  - rótulo vermelho  = gradiente que retropropaga por aquela aresta
  - aresta tracejada = ramo usado só na inferência (`nn`), fora do backward

Gera figuras/exemplos/Exemplo03.png e .svg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
# A derivada de L = e**2 é 2*e. O exemplo3.py do professor escreve
# `grad_e = grad_L * e`, absorvendo o fator 2 na taxa de aprendizado.
# True  -> desenha 2*e (fiel às equações) e adiciona a nota de rodapé
# False -> desenha e (fiel ao código do professor)
FATOR_DOIS = True

SAIDA = Path(__file__).resolve().parent.parent / "figuras" / "exemplos"

# --------------------------------------------------------------------------
# Nós: rótulo None = entrada/parâmetro (círculo); caso contrário, operação (retângulo)
# --------------------------------------------------------------------------
NOS = {
    "x0":    (None,                (0.0,  4.6)),
    "w1":    (None,                (0.0,  3.4)),
    "x1":    (None,                (0.0,  1.0)),
    "w2":    (None,                (0.0, -0.2)),
    "w0":    (None,                (4.6,  4.3)),
    "d":     (None,                (6.5,  0.2)),
    "mul1":  (r"$\times$",         (2.3,  4.0)),
    "mul2":  (r"$\times$",         (2.3,  0.4)),
    "soma1": (r"$+$",              (4.6,  2.2)),
    "soma2": (r"$+$",              (6.5,  2.2)),
    "sub":   (r"$-$",              (8.4,  2.2)),
    "quad":  (r"$(\;)^2$",        (10.1,  2.2)),
    "L":     (r"$L$",             (11.6,  2.2)),
    "sign":  (r"$sign$",           (8.4,  4.6)),
    "nn":    (r"$nn$",            (10.3,  4.6)),
}

# --------------------------------------------------------------------------
# Arestas: (origem, destino, rótulo forward, rótulo do gradiente)
# --------------------------------------------------------------------------
ARESTAS = [
    ("x0",    "mul1",  r"$x_0$",     ""),
    ("w1",    "mul1",  r"$w_1$",     r"$grad\_s0 \cdot x_0$"),
    ("x1",    "mul2",  r"$x_1$",     ""),
    ("w2",    "mul2",  r"$w_2$",     r"$grad\_s1 \cdot x_1$"),
    ("mul1",  "soma1", r"$s_0$",     r"$grad\_s2$"),
    ("mul2",  "soma1", r"$s_1$",     r"$grad\_s2$"),
    ("w0",    "soma2", r"$w_0$",     r"$grad\_y$"),
    ("soma1", "soma2", r"$s_2$",     r"$grad\_y$"),
    ("soma2", "sub",   r"$y$",       r"$grad\_e$"),
    ("d",     "sub",   r"$-\,d$",    ""),
    ("sub",   "quad",  r"$e$",
     r"$2 \cdot e$" if FATOR_DOIS else r"$e$"),
    ("quad",  "L",     r"$L$",       r"$grad\_L = 1$"),
]

# Ramo da inferência: y -> sign -> nn(w, x). Não participa do backward.
ARESTAS_INFERENCIA = [
    ("soma2", "sign",  r"$y$",           ""),
    ("sign",  "nn",    r"$nn(w,x)$",     ""),
]

AZUL = "#33507a"
CINZA = "#8a8a8a"
VERMELHO = "crimson"

# Afastamentos manuais onde os rótulos se sobrepõem.
# lado_fw = -1 joga o rótulo direto para o outro lado da aresta;
# t_fw / t_bw = posição ao longo da aresta (0 = origem, 1 = destino).
AJUSTES = {
    ("d", "sub"):        {"lado_fw": -1},
    ("w0", "soma2"):     {"t_bw": 0.35},
    ("mul1", "soma1"):   {"t_bw": 0.35},
    ("mul2", "soma1"):   {"t_bw": 0.35},
    ("soma2", "sign"):   {"lado_fw": -1, "t_fw": 0.55},
    ("x0", "mul1"):      {"t_fw": 0.45},
    ("w1", "mul1"):      {"t_bw": 0.58},
}


def rotulo_na_aresta(ax, origem, destino, texto, cor, lado, desloc, tamanho, t=0.52):
    """Escreve `texto` ao lado do meio da aresta, deslocado na perpendicular.

    lado = +1 escreve acima da aresta (fluxo direto);
    lado = -1 escreve abaixo (gradiente).
    """
    if not texto:
        return
    (x1, y1), (x2, y2) = NOS[origem][1], NOS[destino][1]
    dx, dy = x2 - x1, y2 - y1
    comprimento = np.hypot(dx, dy)
    normal = (-dy / comprimento, dx / comprimento)
    xm, ym = x1 + t * dx, y1 + t * dy
    ax.text(xm + lado * desloc * normal[0], ym + lado * desloc * normal[1],
            texto, color=cor, fontsize=tamanho, ha="center", va="center",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def main():
    entradas = [n for n, (rot, _) in NOS.items() if rot is None]
    operacoes = [n for n, (rot, _) in NOS.items() if rot is not None]
    pos = {n: p for n, (_, p) in NOS.items()}

    G = nx.DiGraph()
    G.add_nodes_from(NOS)
    G.add_edges_from([(u, v) for u, v, _, _ in ARESTAS])

    H = nx.DiGraph()
    H.add_nodes_from(NOS)
    H.add_edges_from([(u, v) for u, v, _, _ in ARESTAS_INFERENCIA])

    fig, ax = plt.subplots(figsize=(15, 7))

    nx.draw_networkx_nodes(G, pos, nodelist=entradas, node_shape="o",
                           node_size=1500, node_color="#e8eef7",
                           edgecolors=AZUL, linewidths=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=operacoes, node_shape="s",
                           node_size=1700, node_color="white",
                           edgecolors=AZUL, linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=AZUL, width=1.4, arrowsize=15,
                           node_size=1700, min_source_margin=16,
                           min_target_margin=18, ax=ax)
    nx.draw_networkx_edges(H, pos, edge_color=CINZA, width=1.2, arrowsize=15,
                           style="dashed", node_size=1700, min_source_margin=16,
                           min_target_margin=18, ax=ax)

    # entradas mostram o próprio nome; operações, o símbolo da operação
    rotulos = {n: (rot if rot else
                   (f"${n[0]}_{{{n[1:]}}}$" if len(n) > 1 else f"${n}$"))
               for n, (rot, _) in NOS.items()}
    nx.draw_networkx_labels(G, pos, labels=rotulos, font_size=13, ax=ax)

    for origem, destino, forward, gradiente in ARESTAS:
        ajuste = AJUSTES.get((origem, destino), {})
        rotulo_na_aresta(ax, origem, destino, forward, "black",
                         ajuste.get("lado_fw", +1), 0.22, 12,
                         t=ajuste.get("t_fw", 0.52))
        rotulo_na_aresta(ax, origem, destino, gradiente, VERMELHO, -1, 0.26, 10,
                         t=ajuste.get("t_bw", 0.52))

    for origem, destino, forward, _ in ARESTAS_INFERENCIA:
        ajuste = AJUSTES.get((origem, destino), {})
        rotulo_na_aresta(ax, origem, destino, forward, CINZA,
                         ajuste.get("lado_fw", +1), 0.22, 11,
                         t=ajuste.get("t_fw", 0.52))

    ax.set_title(
        "Grafo computacional — exemplo 3:  neurônio  "
        r"$y = w_0 + w_1 x_0 + w_2 x_1$,   $L = (y - d)^2$,   $nn(w,x) = sign(y)$",
        fontsize=14, pad=16)

    legenda = (r"$grad\_w_0 = grad\_y$" + "        "
               r"$grad\_w_1 = grad\_s0 \cdot x_0$" + "        "
               r"$grad\_w_2 = grad\_s1 \cdot x_1$")
    ax.text(0.5, -0.02, legenda, transform=ax.transAxes, ha="center", va="top",
            color=VERMELHO, fontsize=11)

    notas = []
    if FATOR_DOIS:
        notas.append("O exemplo3.py escreve grad_e = grad_L * e, absorvendo o fator 2 na taxa "
                     "de aprendizado.")
    notas.append("O ramo tracejado (sign) só é usado na predição: sua derivada é nula em quase "
                 "todo ponto, por isso o treinamento retropropaga a partir de L.")
    ax.text(0.5, -0.08, "\n".join(notas), transform=ax.transAxes, ha="center", va="top",
            color="#666666", fontsize=8.5)

    ax.margins(0.10)
    ax.axis("off")
    fig.tight_layout()

    SAIDA.mkdir(parents=True, exist_ok=True)
    for extensao in ("png", "svg"):
        destino = SAIDA / f"Exemplo03.{extensao}"
        fig.savefig(destino, dpi=200, bbox_inches="tight")
        print("gerado:", destino)


if __name__ == "__main__":
    main()
