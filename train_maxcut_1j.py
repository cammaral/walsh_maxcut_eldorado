import math
import numpy as np
import matplotlib.pyplot as plt

from maxcut.maxcut import (
    make_graph,
    cut_value,
    index_to_bitstring,
    draw_graph,
    draw_cut_solution,
)
from walsh.quantum_sub import run_subcircuit, draw_subcircuit

# ============================================================
# CONFIG
# ============================================================
N_VERTICES   = 5
PROB_ARESTA  = 0.7
COM_PESO     = True
PESO_MIN     = 1
PESO_MAX     = 3
GRAPH_SEED   = 42
REVERSE_BITS = False

j_train      = 5              # termo U_j que você quer treinar
ER           = 1.0
A_MAX        = math.pi / 4
INIT_BETA    = 0.2
LR           = 10
FD_EPS       = 1e-2
EPOCHS       = 60

SHOW_CIRCUIT = True

# ============================================================
# GRAFO
# ============================================================
G = make_graph(
    n_vertices=N_VERTICES,
    prob_aresta=PROB_ARESTA,
    com_peso=COM_PESO,
    peso_min=PESO_MIN,
    peso_max=PESO_MAX,
    seed=GRAPH_SEED,
)

n = G.number_of_nodes()

# ============================================================
# HAMILTONIANO DIAGONAL DO MAXCUT NO REGISTRADOR
# H = diag(cut_value(x))
# ============================================================
cut_vals = np.zeros(2**n, dtype=float)
bitstrings = []

for idx in range(2**n):
    bits = index_to_bitstring(idx, n, reverse_bits=REVERSE_BITS)
    cut_vals[idx] = float(cut_value(bits, G))
    bitstrings.append("".join(str(int(b)) for b in bits))

H_cut = np.diag(cut_vals)

# ============================================================
# UTILS
# ============================================================
def bounded_coef(beta, a_max=A_MAX):
    return a_max * math.tanh(beta)

def expected_cut_from_state(psi_reg):
    # psi_reg é o estado pós-selecionado do registrador
    psi_reg = np.asarray(psi_reg, dtype=complex)
    exp_val = np.real(np.vdot(psi_reg, H_cut @ psi_reg))
    return float(exp_val)

def evaluate_beta(beta):
    # monta apenas o termo j_train
    avec = np.zeros(2**n, dtype=float)
    avec[j_train] = bounded_coef(beta)

    out = run_subcircuit(
        avec=avec,
        n=n,
        list_j=[j_train],
        er=ER,
        return_full_state=True,
    )

    psi_reg = np.asarray(out["postselected_state"], dtype=complex)
    probs = np.abs(psi_reg)**2
    success_p = float(out["success_probability_ancilla_1"])

    if (success_p < 1e-14) or (not np.all(np.isfinite(probs))) or (probs.sum() < 1e-14):
        probs = np.zeros(2**n, dtype=float)
        expected_cut = 0.0
    else:
        probs = probs / probs.sum()
        expected_cut = expected_cut_from_state(psi_reg)

    loss = -expected_cut
    return {
        "beta": float(beta),
        "coef": float(bounded_coef(beta)),
        "loss": float(loss),
        "expected_cut": float(expected_cut),
        "success_probability": float(success_p),
        "probs": probs,
        "psi_reg": psi_reg,
        "avec": avec,
    }

# ============================================================
# TREINO
# ============================================================
beta = INIT_BETA
history_loss = []
history_coef = []
history_expected = []
history_success = []

best_loss = float("inf")
best_data = None

for epoch in range(EPOCHS):
    eval_plus  = evaluate_beta(beta + FD_EPS)
    eval_minus = evaluate_beta(beta - FD_EPS)
    grad = (eval_plus["loss"] - eval_minus["loss"]) / (2.0 * FD_EPS)

    beta = beta - LR * grad

    data = evaluate_beta(beta)

    history_loss.append(data["loss"])
    history_coef.append(data["coef"])
    history_expected.append(data["expected_cut"])
    history_success.append(data["success_probability"])

    if data["loss"] < best_loss:
        best_loss = data["loss"]
        best_data = data.copy()

    print(
        f"epoch={epoch:03d} | "
        f"beta={data['beta']:+.6f} | "
        f"a_j={data['coef']:+.6f} | "
        f"loss={data['loss']:.6f} | "
        f"<H_cut>={data['expected_cut']:.6f} | "
        f"p_succ={data['success_probability']:.6f}"
    )

# ============================================================
# RESULTADO FINAL
# ============================================================
final_probs = best_data["probs"]
best_idx = int(np.argmax(final_probs)) if final_probs.sum() > 0 else 0
best_bits = index_to_bitstring(best_idx, n, reverse_bits=REVERSE_BITS)
best_bitstring = "".join(str(int(b)) for b in best_bits)
best_prob = float(final_probs[best_idx])
best_cut = float(cut_vals[best_idx])

print("\n================ RESULTADO FINAL ================")
print("j treinado                   =", j_train)
print("beta final                   =", best_data["beta"])
print("coeficiente final            =", best_data["coef"])
print("loss final                   =", best_data["loss"])
print("expected cut final           =", best_data["expected_cut"])
print("bitstring mais provável      =", best_bitstring)
print("probabilidade dessa bitstring=", best_prob)
print("cut dessa bitstring          =", best_cut)

# ============================================================
# PLOTS
# ============================================================

# 1) loss
plt.figure(figsize=(7, 4))
plt.plot(history_loss, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss = -<H_cut>")
plt.title(f"Treino de U_{j_train}: loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2) coeficiente
plt.figure(figsize=(7, 4))
plt.plot(history_coef, marker="o")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel(f"a_{j_train}")
plt.title(f"Treino de U_{j_train}: coeficiente")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3) expected cut
plt.figure(figsize=(7, 4))
plt.plot(history_expected, marker="o")
plt.xlabel("Epoch")
plt.ylabel("<H_cut>")
plt.title(f"Treino de U_{j_train}: valor esperado do MaxCut")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4) probabilidades finais
plt.figure(figsize=(10, 4))
plt.bar(range(2**n), final_probs)
plt.xlabel("Índice da bitstring")
plt.ylabel("Probabilidade")
plt.title(f"Distribuição final para U_{j_train}")
plt.tight_layout()
plt.show()

# 5) desenha o grafo
draw_graph(G, title="Grafo do problema MaxCut")
plt.show()

# 6) desenha a solução mais provável
draw_cut_solution(
    G,
    best_bits,
    title=f"Solução mais provável | cut={best_cut:.3f} | prob={best_prob:.4f}"
)
plt.show()

# 7) desenha o circuito
if SHOW_CIRCUIT:
    fig, ax = draw_subcircuit(best_data["avec"], n, [j_train], er=ER)
    plt.show()