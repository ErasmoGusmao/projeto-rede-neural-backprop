# -*- coding: utf-8 -*-
"""
Grafo computacional da rede proposta (seção 6 do notebook): 2 -> 4 -> 1, tanh na camada
oculta e sigmoide na saída, perda L = 0.5 * e**2.

Os nomes dos nós e das arestas são exatamente os do esqueleto das funções `forward` e
`backward` das seções 6.4 e 6.5 (parâmetros no dicionário `params`: W1 (4, 2), b1 (4,), W2 (4,), b2 (1,)).

Forward (`forward`), neurônio oculto i = 0..3:

    s1_i0 = W1[i, 0] * x[0]
    s1_i1 = W1[i, 1] * x[1]
    v1_i  = s1_i0 + s1_i1 + b1[i]
    y1_i  = tanh(v1_i)

    s2_i  = W2[i] * y1_i
    v2    = s2_0 + s2_1 + s2_2 + s2_3 + b2[0]
    y2    = sigmoid(v2)
    e     = y2 - d
    L     = 0.5 * e**2

Backward (`backward`), na ordem inversa:

    grad_L  = 1
    grad_e  = grad_L * e                         # (1)
    grad_y2 = grad_e                             # (2)
    grad_v2 = grad_y2 * y2 * (1 - y2)            # (3)
    grad_b2[0] = grad_v2                         # (4)
    para cada i:
        grad_s2_i     = grad_v2                  #     a soma repassa o gradiente
        grad_W2[i]    = grad_s2_i * y1_i         # (4)
        grad_y1_i     = grad_s2_i * W2[i]        # (5)
        grad_v1_i     = grad_y1_i * (1 - y1_i**2)# (6)
        grad_b1[i]    = grad_v1_i                # (7)
        grad_s1_i0    = grad_v1_i ;  grad_s1_i1 = grad_v1_i
        grad_W1[i, 0] = grad_s1_i0 * x[0]        # (7)
        grad_W1[i, 1] = grad_s1_i1 * x[1]        # (7)

Convenções (as mesmas de gera_grafo_baseline.py e gera_grafo_exemplo03.py):
  - nó redondo       = variável de entrada ou parâmetro
  - nó retangular    = operação elementar
  - rótulo preto     = nome da variável do código (fluxo direto)
  - rótulo vermelho  = gradiente que retropropaga por aquela aresta

Gera, em figuras/grafos/:
  grafo_proposta_2-4-1_forward.{png,svg}   -> só o forward (não usado no notebook)
  grafo_proposta_2-4-1_backward.{png,svg}  -> seção 6.3 do notebook (backward anotado)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

SAIDA = Path(__file__).resolve().parent.parent / "figuras" / "grafos"

AZUL = "#33507a"
VERMELHO = "crimson"

N_OCULTOS = 4
Y_CENTRO = [12.0, 8.0, 4.0, 0.0]        # altura do centro de cada neurônio oculto
Y_MUL2 = [10.5, 7.5, 4.5, 1.5]          # altura dos produtos W2[i] * y1_i

# --------------------------------------------------------------------------
# Nós: nome -> (rótulo, posição, tipo)   tipo: "in" = entrada/parâmetro, "op" = operação
# --------------------------------------------------------------------------
NOS = {
    "x0": (r"$x_0$", (0.6, 8.0), "in"),
    "x1": (r"$x_1$", (0.6, 4.0), "in"),
    "b2": (r"$b2_0$", (12.4, 9.0), "in"),
    "d":  (r"$d$",    (16.0, 4.0), "in"),
    "soma2": (r"$+$",      (12.4, 6.0), "op"),
    "sig2":  (r"$\sigma$", (14.2, 6.0), "op"),
    "sub":   (r"$-$",      (16.0, 6.0), "op"),
    "quad":  (r"$\frac{1}{2}(\;)^2$", (17.8, 6.0), "op"),
    "L":     (r"$L$",      (19.4, 6.0), "op"),
}
for i in range(N_OCULTOS):
    yc, ym = Y_CENTRO[i], Y_MUL2[i]
    NOS[f"W1_{i}0"] = (rf"$W1_{{{i}0}}$", (-1.6, yc + 0.9), "in")
    NOS[f"W1_{i}1"] = (rf"$W1_{{{i}1}}$", (-1.6, yc - 0.9), "in")
    NOS[f"b1_{i}"]  = (rf"$b1_{i}$",      (4.6, yc + 2.0), "in")
    NOS[f"W2_{i}"]  = (rf"$W2_{i}$",      (8.8, ym - 1.7), "in")
    NOS[f"mul1_{i}0"] = (r"$\times$", (2.8, yc + 0.9), "op")
    NOS[f"mul1_{i}1"] = (r"$\times$", (2.8, yc - 0.9), "op")
    NOS[f"soma1_{i}"] = (r"$+$",      (4.6, yc), "op")
    NOS[f"tanh_{i}"]  = (r"$\tanh$",  (6.4, yc), "op")
    NOS[f"mul2_{i}"]  = (r"$\times$", (10.4, ym), "op")

# --------------------------------------------------------------------------
# Arestas: (origem, destino, rótulo forward, rótulo do gradiente)
# --------------------------------------------------------------------------
ARESTAS = []
for i in range(N_OCULTOS):
    ARESTAS += [
        (f"W1_{i}0", f"mul1_{i}0", rf"$W1_{{{i}0}}$", rf"$grad\_s1\_{i}0 \cdot x_0$"),
        ("x0",       f"mul1_{i}0", r"$x_0$",           ""),
        (f"W1_{i}1", f"mul1_{i}1", rf"$W1_{{{i}1}}$", rf"$grad\_s1\_{i}1 \cdot x_1$"),
        ("x1",       f"mul1_{i}1", r"$x_1$",           ""),
        (f"mul1_{i}0", f"soma1_{i}", rf"$s1\_{i}0$",   rf"$grad\_s1\_{i}0$"),
        (f"mul1_{i}1", f"soma1_{i}", rf"$s1\_{i}1$",   rf"$grad\_s1\_{i}1$"),
        (f"b1_{i}",    f"soma1_{i}", rf"$b1_{i}$",     rf"$grad\_v1\_{i}$"),
        (f"soma1_{i}", f"tanh_{i}",  rf"$v1\_{i}$",    rf"$grad\_v1\_{i}$"),
        (f"tanh_{i}",  f"mul2_{i}",  rf"$y1\_{i}$",    rf"$grad\_y1\_{i}$"),
        (f"W2_{i}",    f"mul2_{i}",  rf"$W2_{i}$",     rf"$grad\_s2\_{i} \cdot y1\_{i}$"),
        (f"mul2_{i}",  "soma2",      rf"$s2\_{i}$",    rf"$grad\_s2\_{i}$"),
    ]
ARESTAS += [
    ("b2",    "soma2", r"$b2_0$",  r"$grad\_v2$"),
    ("soma2", "sig2",  r"$v2$",    r"$grad\_v2$"),
    ("sig2",  "sub",   r"$y2$",    r"$grad\_y2$"),
    ("d",     "sub",   r"$-\,d$",  ""),
    ("sub",   "quad",  r"$e$",     r"$grad\_e = e$"),
    ("quad",  "L",     r"$L$",     r"$grad\_L = 1$"),
]

# Afastamentos manuais onde os rótulos se sobrepõem.
# lado_fw / lado_bw = de que lado da aresta fica o rótulo direto / do gradiente
# (+1 é o lado da normal à esquerda do sentido da aresta; padrão: direto +1, gradiente -1);
# t_fw / t_bw = posição ao longo da aresta (0 = origem, 1 = destino).
AJUSTES = {
    ("b2", "soma2"): {"lado_fw": +1, "lado_bw": +1, "t_fw": 0.30, "t_bw": 0.70},
    ("d",  "sub"):   {"lado_fw": -1},
}
for i in range(N_OCULTOS):
    # aresta curta e quase horizontal: rótulo no meio; aresta longa e inclinada: perto do destino
    dy0, dy1 = Y_CENTRO[i] + 0.9 - 8.0, Y_CENTRO[i] - 0.9 - 4.0
    AJUSTES[("x0", f"mul1_{i}0")] = {"t_fw": 0.50 if abs(dy0) < 2 else 0.80, "lado_fw": +1 if dy0 > 0 else -1}
    AJUSTES[("x1", f"mul1_{i}1")] = {"t_fw": 0.50 if abs(dy1) < 2 else 0.80, "lado_fw": +1 if dy1 > 0 else -1}
    AJUSTES[(f"W1_{i}0", f"mul1_{i}0")] = {"t_fw": 0.40, "t_bw": 0.35}
    AJUSTES[(f"W1_{i}1", f"mul1_{i}1")] = {"t_fw": 0.40, "t_bw": 0.35}
    AJUSTES[(f"mul1_{i}0", f"soma1_{i}")] = {"t_bw": 0.40}
    AJUSTES[(f"mul1_{i}1", f"soma1_{i}")] = {"t_bw": 0.40}
    AJUSTES[(f"b1_{i}", f"soma1_{i}")] = {"t_fw": 0.35, "t_bw": 0.35}
    AJUSTES[(f"tanh_{i}", f"mul2_{i}")] = {"t_bw": 0.35}
    AJUSTES[(f"W2_{i}", f"mul2_{i}")] = {"t_fw": 0.45, "t_bw": 0.45}
    AJUSTES[(f"mul2_{i}", "soma2")] = {"t_fw": 0.50, "t_bw": 0.40}


def rotulo_na_aresta(ax, origem, destino, texto, cor, lado, desloc, tamanho, t=0.52):
    """Escreve `texto` ao lado do meio da aresta, deslocado na perpendicular.

    lado = +1 escreve de um lado da aresta (fluxo direto);
    lado = -1 escreve do outro (gradiente).
    """
    if not texto:
        return
    (x1, y1), (x2, y2) = NOS[origem][1], NOS[destino][1]
    dx, dy = x2 - x1, y2 - y1
    comprimento = np.hypot(dx, dy)
    normal = (-dy / comprimento, dx / comprimento)
    xm, ym = x1 + t * dx, y1 + t * dy
    ox, oy = lado * desloc * normal[0], lado * desloc * normal[1]
    # em aresta quase vertical o deslocamento é horizontal: ancorar o texto pela
    # borda voltada para a aresta, para que ele não a atravesse
    ha = "center" if abs(ox) <= abs(oy) else ("left" if ox > 0 else "right")
    ax.text(xm + ox, ym + oy,
            texto, color=cor, fontsize=tamanho, ha=ha, va="center",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def desenha(com_backward):
    entradas = [n for n, (_, _, tipo) in NOS.items() if tipo == "in"]
    operacoes = [n for n, (_, _, tipo) in NOS.items() if tipo == "op"]
    pos = {n: p for n, (_, p, _) in NOS.items()}
    rotulos = {n: rot for n, (rot, _, _) in NOS.items()}

    G = nx.DiGraph()
    G.add_nodes_from(NOS)
    G.add_edges_from([(u, v) for u, v, _, _ in ARESTAS])

    fig, ax = plt.subplots(figsize=(23, 17))

    nx.draw_networkx_nodes(G, pos, nodelist=entradas, node_shape="o",
                           node_size=1500, node_color="#e8eef7",
                           edgecolors=AZUL, linewidths=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=operacoes, node_shape="s",
                           node_size=1700, node_color="white",
                           edgecolors=AZUL, linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=AZUL, width=1.4, arrowsize=15,
                           node_size=1700, min_source_margin=16,
                           min_target_margin=18, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=rotulos, font_size=13, ax=ax)

    for origem, destino, forward, gradiente in ARESTAS:
        ajuste = AJUSTES.get((origem, destino), {})
        rotulo_na_aresta(ax, origem, destino, forward, "black",
                         ajuste.get("lado_fw", +1), 0.24, 12,
                         t=ajuste.get("t_fw", 0.52))
        if com_backward:
            rotulo_na_aresta(ax, origem, destino, gradiente, VERMELHO,
                             ajuste.get("lado_bw", -1), 0.28, 10,
                             t=ajuste.get("t_bw", 0.52))

    qual = "backward anotado" if com_backward else "forward"
    ax.set_title(
        f"Grafo computacional — rede proposta 2→4→1 ({qual}):  "
        r"$y1_i = \tanh(W1_{i0}\,x_0 + W1_{i1}\,x_1 + b1_i)$,   "
        r"$y2 = \sigma\left(\sum_i W2_i\,y1_i + b2_0\right)$,   "
        r"$L = \frac{1}{2}(y2 - d)^2$",
        fontsize=14, pad=16)

    if com_backward:
        legenda = "\n".join([
            r"$grad\_v2 = grad\_y2 \cdot y2\,(1-y2)$      "
            r"$grad\_b2_0 = grad\_v2$      "
            r"$grad\_s2\_i = grad\_v2$      "
            r"$grad\_W2_i = grad\_s2\_i \cdot y1\_i$      "
            r"$grad\_y1\_i = grad\_s2\_i \cdot W2_i$",
            r"$grad\_v1\_i = grad\_y1\_i \cdot (1 - y1\_i^{\,2})$      "
            r"$grad\_b1_i = grad\_v1\_i$      "
            r"$grad\_s1\_ij = grad\_v1\_i$      "
            r"$grad\_W1_{ij} = grad\_s1\_ij \cdot x_j$",
        ])
        ax.text(0.5, -0.02, legenda, transform=ax.transAxes, ha="center", va="top",
                color=VERMELHO, fontsize=11, linespacing=1.6)
        nota = ("Rótulo preto = variável de forward; rótulo vermelho = gradiente que retropropaga "
                "pela aresta em backward (prefixo grad_).  As arestas que saem de x0 e x1 não têm "
                "rótulo vermelho: as entradas não são parâmetros, e o gradiente delas — a soma de "
                "quatro contribuições grad_s1_ij · W1[i, j] — não é calculado.")
        ax.text(0.5, -0.07, nota, transform=ax.transAxes, ha="center", va="top",
                color="#666666", fontsize=8.5)
    else:
        nota = ("Cada nó retangular é uma linha de forward; o rótulo da aresta é o nome da "
                "variável que sai dele. O backward anotado está no grafo seguinte.")
        ax.text(0.5, -0.02, nota, transform=ax.transAxes, ha="center", va="top",
                color="#666666", fontsize=9)

    ax.margins(0.05)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    for com_backward, sufixo in ((False, "forward"), (True, "backward")):
        fig = desenha(com_backward)
        for extensao in ("png", "svg"):
            destino = SAIDA / f"grafo_proposta_2-4-1_{sufixo}.{extensao}"
            fig.savefig(destino, dpi=150, bbox_inches="tight")
            print("gerado:", destino)
        plt.close(fig)


if __name__ == "__main__":
    main()
