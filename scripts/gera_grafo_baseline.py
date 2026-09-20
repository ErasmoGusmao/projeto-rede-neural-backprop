# -*- coding: utf-8 -*-
"""
Grafo computacional do baseline (seção 6 do notebook): rede 2 -> 2 -> 1 com sigmoide
em todas as camadas, reprodução do exemplo4.py do professor.

O desenho segue, operação por operação, as funções `forward_baseline` e
`backward_baseline` da seção 6 do notebook — e não o exemplo4.py original. A diferença
é que a soma de cada neurônio é feita numa única operação (v0 = s00 + s01 + b0[0]),
sem os nós intermediários s02/s12/s22 do script do professor.

Forward (`forward_baseline`):

    s00 = w0[0,0] * x[0]        s10 = w0[1,0] * x[0]        s20 = w1[0] * y0
    s01 = w0[0,1] * x[1]        s11 = w0[1,1] * x[1]        s21 = w1[1] * y1
    v0  = s00 + s01 + b0[0]     v1  = s10 + s11 + b0[1]     v2  = s20 + s21 + b1[0]
    y0  = sigmoid(v0)           y1  = sigmoid(v1)           y2  = sigmoid(v2)

    e = y2 - d
    L = 0.5 * e**2

Backward (`backward_baseline`), na ordem inversa:

    grad_L  = 1
    grad_e  = grad_L * e
    grad_y2 = grad_e
    grad_v2 = grad_y2 * y2 * (1 - y2)
    grad_b1[0] = grad_v2
    grad_w1[0] = grad_v2 * y0      grad_y0 = grad_v2 * w1[0]
    grad_w1[1] = grad_v2 * y1      grad_y1 = grad_v2 * w1[1]
    grad_v0 = grad_y0 * y0 * (1 - y0)
    grad_v1 = grad_y1 * y1 * (1 - y1)
    grad_b0[0] = grad_v0           grad_b0[1] = grad_v1
    grad_w0[0,0] = grad_v0 * x[0]  grad_w0[0,1] = grad_v0 * x[1]
    grad_w0[1,0] = grad_v1 * x[0]  grad_w0[1,1] = grad_v1 * x[1]

Convenções (as mesmas de scripts/gera_grafo_exemplo03.py):
  - nó redondo       = variável de entrada ou parâmetro
  - nó retangular    = operação elementar
  - rótulo preto     = nome da variável do código (fluxo direto)
  - rótulo vermelho  = gradiente que retropropaga por aquela aresta

Gera figuras/grafos/grafo_baseline_2-2-1.png e .svg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

SAIDA = Path(__file__).resolve().parent.parent / "figuras" / "grafos"

AZUL = "#33507a"
VERMELHO = "crimson"

# --------------------------------------------------------------------------
# Nós: nome -> (rótulo, posição, tipo)   tipo: "in" = entrada/parâmetro, "op" = operação
# --------------------------------------------------------------------------
NOS = {
    # ---- parâmetros da camada oculta (w0, b0) e entradas ----
    "w000":  (r"$w0_{00}$",   (-1.6,  9.2), "in"),
    "w001":  (r"$w0_{01}$",   (-1.6,  7.4), "in"),
    "x0":    (r"$x_0$",       ( 0.4,  5.9), "in"),
    "x1":    (r"$x_1$",       ( 0.4,  3.5), "in"),
    "w010":  (r"$w0_{10}$",   (-1.6,  2.0), "in"),
    "w011":  (r"$w0_{11}$",   (-1.6,  0.2), "in"),
    "b00":   (r"$b0_{0}$",    ( 4.6, 10.6), "in"),
    "b01":   (r"$b0_{1}$",    ( 4.6, -1.2), "in"),
    # ---- parâmetros da camada de saída (w1, b1) e alvo ----
    "w10":   (r"$w1_{0}$",    ( 8.4,  8.8), "in"),
    "w11":   (r"$w1_{1}$",    ( 8.4,  0.6), "in"),
    "b10":   (r"$b1_{0}$",    (10.4,  7.4), "in"),
    "d":     (r"$d$",         (14.0,  2.6), "in"),
    # ---- neurônio 0 da camada oculta ----
    "mul00": (r"$\times$",    ( 2.8,  9.2), "op"),
    "mul01": (r"$\times$",    ( 2.8,  7.4), "op"),
    "soma0": (r"$+$",         ( 4.6,  8.3), "op"),
    "sig0":  (r"$\sigma$",    ( 6.4,  8.3), "op"),
    # ---- neurônio 1 da camada oculta ----
    "mul10": (r"$\times$",    ( 2.8,  2.0), "op"),
    "mul11": (r"$\times$",    ( 2.8,  0.2), "op"),
    "soma1": (r"$+$",         ( 4.6,  1.1), "op"),
    "sig1":  (r"$\sigma$",    ( 6.4,  1.1), "op"),
    # ---- neurônio de saída ----
    "mul20": (r"$\times$",    ( 8.4,  6.8), "op"),
    "mul21": (r"$\times$",    ( 8.4,  2.6), "op"),
    "soma2": (r"$+$",         (10.4,  4.7), "op"),
    "sig2":  (r"$\sigma$",    (12.2,  4.7), "op"),
    # ---- erro e perda ----
    "sub":   (r"$-$",         (14.0,  4.7), "op"),
    "quad":  (r"$\frac{1}{2}(\;)^2$", (15.8, 4.7), "op"),
    "L":     (r"$L$",         (17.4,  4.7), "op"),
}

# --------------------------------------------------------------------------
# Arestas: (origem, destino, rótulo forward, rótulo do gradiente)
# --------------------------------------------------------------------------
ARESTAS = [
    # neurônio 0 da camada oculta
    ("w000",  "mul00", r"$w0_{00}$", r"$grad\_v0 \cdot x_0$"),
    ("x0",    "mul00", r"$x_0$",     ""),
    ("w001",  "mul01", r"$w0_{01}$", r"$grad\_v0 \cdot x_1$"),
    ("x1",    "mul01", r"$x_1$",     ""),
    ("mul00", "soma0", r"$s_{00}$",  r"$grad\_v0$"),
    ("mul01", "soma0", r"$s_{01}$",  r"$grad\_v0$"),
    ("b00",   "soma0", r"$b0_0$",    r"$grad\_v0$"),
    ("soma0", "sig0",  r"$v_0$",     r"$grad\_v0$"),
    ("sig0",  "mul20", r"$y_0$",     r"$grad\_y0$"),
    # neurônio 1 da camada oculta
    ("w010",  "mul10", r"$w0_{10}$", r"$grad\_v1 \cdot x_0$"),
    ("x0",    "mul10", r"$x_0$",     ""),
    ("w011",  "mul11", r"$w0_{11}$", r"$grad\_v1 \cdot x_1$"),
    ("x1",    "mul11", r"$x_1$",     ""),
    ("mul10", "soma1", r"$s_{10}$",  r"$grad\_v1$"),
    ("mul11", "soma1", r"$s_{11}$",  r"$grad\_v1$"),
    ("b01",   "soma1", r"$b0_1$",    r"$grad\_v1$"),
    ("soma1", "sig1",  r"$v_1$",     r"$grad\_v1$"),
    ("sig1",  "mul21", r"$y_1$",     r"$grad\_y1$"),
    # neurônio de saída
    ("w10",   "mul20", r"$w1_0$",    r"$grad\_v2 \cdot y_0$"),
    ("w11",   "mul21", r"$w1_1$",    r"$grad\_v2 \cdot y_1$"),
    ("mul20", "soma2", r"$s_{20}$",  r"$grad\_v2$"),
    ("mul21", "soma2", r"$s_{21}$",  r"$grad\_v2$"),
    ("b10",   "soma2", r"$b1_0$",    r"$grad\_v2$"),
    ("soma2", "sig2",  r"$v_2$",     r"$grad\_v2$"),
    ("sig2",  "sub",   r"$y_2$",     r"$grad\_y2$"),
    ("d",     "sub",   r"$-\,d$",    ""),
    # erro e perda
    ("sub",   "quad",  r"$e$",       r"$grad\_e = e$"),
    ("quad",  "L",     r"$L$",       r"$grad\_L = 1$"),
]

# Afastamentos manuais onde os rótulos se sobrepõem.
# lado_fw / lado_bw = de que lado da aresta fica o rótulo direto / do gradiente
# (+1 é o lado da normal à esquerda do sentido da aresta; padrão: direto +1, gradiente -1);
# t_fw / t_bw = posição ao longo da aresta (0 = origem, 1 = destino).
AJUSTES = {
    # entradas: rótulo perto do nó de multiplicação, fora da região de cruzamento
    ("x0",    "mul00"): {"t_fw": 0.55},
    ("x1",    "mul01"): {"t_fw": 0.65, "lado_fw": -1},
    ("x0",    "mul10"): {"t_fw": 0.65, "lado_fw": -1},
    ("x1",    "mul11"): {"t_fw": 0.55, "lado_fw": -1},
    # pesos da camada oculta: gradiente antes do ponto em que a aresta de x_j cruza
    ("w000",  "mul00"): {"t_fw": 0.40, "t_bw": 0.35},
    ("w001",  "mul01"): {"t_fw": 0.40, "t_bw": 0.35},
    ("w010",  "mul10"): {"t_fw": 0.40, "t_bw": 0.35},
    ("w011",  "mul11"): {"t_fw": 0.40, "t_bw": 0.35},
    ("mul00", "soma0"): {"t_bw": 0.40},
    ("mul01", "soma0"): {"t_bw": 0.40},
    ("mul10", "soma1"): {"t_bw": 0.40},
    ("mul11", "soma1"): {"t_bw": 0.40},
    # arestas verticais descendentes: lado +1 = direita, -1 = esquerda
    ("b00",   "soma0"): {"t_bw": 0.40},
    ("w10",   "mul20"): {"lado_fw": -1, "lado_bw": +1, "t_fw": 0.35, "t_bw": 0.50},
    ("b10",   "soma2"): {"lado_fw": -1, "lado_bw": +1, "t_fw": 0.30, "t_bw": 0.60},
    # arestas verticais ascendentes: lado +1 = esquerda, -1 = direita (padrão já serve)
    ("w11",   "mul21"): {"t_fw": 0.35, "t_bw": 0.50},
    ("b01",   "soma1"): {"t_bw": 0.40},
    ("mul20", "soma2"): {"t_bw": 0.40},
    ("mul21", "soma2"): {"t_bw": 0.40},
    ("d",     "sub"):   {"lado_fw": -1},
}


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


def main():
    entradas = [n for n, (_, _, tipo) in NOS.items() if tipo == "in"]
    operacoes = [n for n, (_, _, tipo) in NOS.items() if tipo == "op"]
    pos = {n: p for n, (_, p, _) in NOS.items()}
    rotulos = {n: rot for n, (rot, _, _) in NOS.items()}

    G = nx.DiGraph()
    G.add_nodes_from(NOS)
    G.add_edges_from([(u, v) for u, v, _, _ in ARESTAS])

    fig, ax = plt.subplots(figsize=(21, 10))

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
        rotulo_na_aresta(ax, origem, destino, gradiente, VERMELHO,
                         ajuste.get("lado_bw", -1), 0.28, 10,
                         t=ajuste.get("t_bw", 0.52))

    ax.set_title(
        "Grafo computacional — baseline 2→2→1 (exemplo4.py):  "
        r"$y_2 = \sigma(w1_0\,y_0 + w1_1\,y_1 + b1_0)$,   "
        r"$y_i = \sigma(w0_{i0}\,x_0 + w0_{i1}\,x_1 + b0_i)$,   "
        r"$L = \frac{1}{2}(y_2 - d)^2$",
        fontsize=14, pad=16)

    legenda = "\n".join([
        r"$grad\_v2 = grad\_y2 \cdot y_2(1-y_2)$      "
        r"$grad\_y0 = grad\_v2 \cdot w1_0$      "
        r"$grad\_y1 = grad\_v2 \cdot w1_1$      "
        r"$grad\_b1_0 = grad\_v2$      "
        r"$grad\_w1_i = grad\_v2 \cdot y_i$",
        r"$grad\_v0 = grad\_y0 \cdot y_0(1-y_0)$      "
        r"$grad\_v1 = grad\_y1 \cdot y_1(1-y_1)$      "
        r"$grad\_b0_i = grad\_vi$      "
        r"$grad\_w0_{ij} = grad\_vi \cdot x_j$",
    ])
    ax.text(0.5, -0.02, legenda, transform=ax.transAxes, ha="center", va="top",
            color=VERMELHO, fontsize=11, linespacing=1.6)

    nota = ("Rótulo preto = variável de forward_baseline; rótulo vermelho = gradiente que "
            "retropropaga pela aresta em backward_baseline (prefixo grad_).  "
            "Como L = ½e², grad_e = e sem fator 2.")
    ax.text(0.5, -0.10, nota, transform=ax.transAxes, ha="center", va="top",
            color="#666666", fontsize=8.5)

    ax.margins(0.06)
    ax.axis("off")
    fig.tight_layout()

    SAIDA.mkdir(parents=True, exist_ok=True)
    for extensao in ("png", "svg"):
        destino = SAIDA / f"grafo_baseline_2-2-1.{extensao}"
        fig.savefig(destino, dpi=200, bbox_inches="tight")
        print("gerado:", destino)


if __name__ == "__main__":
    main()
