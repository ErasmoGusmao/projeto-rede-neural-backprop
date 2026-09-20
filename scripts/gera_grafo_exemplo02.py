# -*- coding: utf-8 -*-
"""
Grafo computacional do exemplo 2 (interpolação polinomial).

Equações do exemplo02.ipynb:

    p(x, w) = w0 + w1 * x + w2 * x**2
    L       = (p(x, w) - y)**2

Decompostas nas operações elementares do `df` do exemplo02.ipynb:

    s00 = x0 * w1
    s01 = x0**2 * w2
    s02 = w0 + s00 + s01 - y0
    L   = s02**2

Convenções (seção 2.4 do notebook do projeto):
  - nó redondo     = variável de entrada ou parâmetro
  - nó retangular  = operação elementar
  - rótulo preto   = nome da variável do código (fluxo direto)
  - rótulo vermelho = gradiente que retropropaga por aquela aresta

Gera figuras/exemplos/Exemplo02.png e .svg
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
# A derivada de L = s02**2 é 2*s02. O exemplo2.py do professor escreve
# `grad_s02 = grad_l * s02`, absorvendo o fator 2 na taxa de aprendizado.
# True  -> desenha 2*s02 (fiel às equações) e adiciona a nota de rodapé
# False -> desenha s02 (fiel ao código do professor)
FATOR_DOIS = True

SAIDA = Path(__file__).resolve().parent.parent / "figuras" / "exemplos"

# --------------------------------------------------------------------------
# Nós: rótulo None = entrada/parâmetro (círculo); caso contrário, operação (retângulo)
# --------------------------------------------------------------------------
NOS = {
    "x0":   (None,               (0.0,  1.5)),
    "w0":   (None,               (2.3,  4.1)),
    "w1":   (None,               (0.0,  3.1)),
    "w2":   (None,               (0.0, -0.1)),
    "y0":   (None,               (2.3, -1.1)),
    "mul1": (r"$\times$",        (2.3,  2.6)),
    "mul2": (r"$\times$",        (2.3,  0.4)),
    "soma": (r"$+$",             (4.6,  1.5)),
    "quad": (r"$(\;)^2$",        (6.4,  1.5)),
    "L":    (r"$L$",             (8.0,  1.5)),
}

# --------------------------------------------------------------------------
# Arestas: (origem, destino, rótulo forward, rótulo do gradiente)
# --------------------------------------------------------------------------
ARESTAS = [
    ("x0",   "mul1", r"$x_0$",       ""),
    ("w1",   "mul1", r"$w_1$",       r"$grad\_s00 \cdot x_0$"),
    ("x0",   "mul2", r"$x_0^2$",     ""),
    ("w2",   "mul2", r"$w_2$",       r"$grad\_s01 \cdot x_0^2$"),
    ("w0",   "soma", r"$w_0$",       r"$grad\_s02$"),
    ("mul1", "soma", r"$s_{00}$",    r"$grad\_s02$"),
    ("mul2", "soma", r"$s_{01}$",    r"$grad\_s02$"),
    ("y0",   "soma", r"$-\,y_0$",    ""),
    ("soma", "quad", r"$s_{02}$",
     r"$2 \cdot s_{02}$" if FATOR_DOIS else r"$s_{02}$"),
    ("quad", "L",    r"$L$",         r"$1$"),
]

AZUL = "#33507a"
VERMELHO = "crimson"


# Afastamentos manuais onde os rótulos se sobrepõem perto do nó de soma.
# lado_fw = -1 joga o rótulo direto para o outro lado da aresta;
# t_fw / t_bw = posição ao longo da aresta (0 = origem, 1 = destino).
AJUSTES = {
    ("y0", "soma"):   {"lado_fw": -1},
    ("w0", "soma"):   {"t_bw": 0.33},
    ("mul1", "soma"): {"t_bw": 0.34},
    ("mul2", "soma"): {"t_bw": 0.34},
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

    fig, ax = plt.subplots(figsize=(13, 6.5))

    nx.draw_networkx_nodes(G, pos, nodelist=entradas, node_shape="o",
                           node_size=1500, node_color="#e8eef7",
                           edgecolors=AZUL, linewidths=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=operacoes, node_shape="s",
                           node_size=1700, node_color="white",
                           edgecolors=AZUL, linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=AZUL, width=1.4, arrowsize=15,
                           node_size=1700, min_source_margin=16,
                           min_target_margin=18, ax=ax)

    # entradas mostram o próprio nome; operações, o símbolo da operação
    rotulos = {n: (rot if rot else f"${n[0]}_{{{n[1:]}}}$") for n, (rot, _) in NOS.items()}
    nx.draw_networkx_labels(G, pos, labels=rotulos, font_size=13, ax=ax)

    for origem, destino, forward, gradiente in ARESTAS:
        ajuste = AJUSTES.get((origem, destino), {})
        rotulo_na_aresta(ax, origem, destino, forward, "black",
                         ajuste.get("lado_fw", +1), 0.22, 12,
                         t=ajuste.get("t_fw", 0.52))
        rotulo_na_aresta(ax, origem, destino, gradiente, VERMELHO, -1, 0.26, 10,
                         t=ajuste.get("t_bw", 0.52))

    ax.set_title(
        r"Grafo computacional — exemplo 2:  "
        r"$p(x,w) = w_0 + w_1 x + w_2 x^2$,   $L = \left(p(x,w) - y\right)^2$",
        fontsize=14, pad=16)

    legenda = (r"$grad\_w_0 = grad\_s02$" + "        "
               r"$grad\_w_1 = grad\_s02 \cdot x_0$" + "        "
               r"$grad\_w_2 = grad\_s02 \cdot x_0^2$")
    ax.text(0.5, -0.02, legenda, transform=ax.transAxes, ha="center", va="top",
            color=VERMELHO, fontsize=11)

    if FATOR_DOIS:
        ax.text(0.5, -0.09,
                "O exemplo2.py escreve grad_s02 = grad_l * s02, absorvendo o fator 2 na taxa "
                "de aprendizado.",
                transform=ax.transAxes, ha="center", va="top",
                color="#666666", fontsize=8.5)

    ax.margins(0.10)
    ax.axis("off")
    fig.tight_layout()

    SAIDA.mkdir(parents=True, exist_ok=True)
    for extensao in ("png", "svg"):
        destino = SAIDA / f"Exemplo02.{extensao}"
        fig.savefig(destino, dpi=200, bbox_inches="tight")
        print("gerado:", destino)


if __name__ == "__main__":
    main()
