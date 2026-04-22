from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from maxcut.maxcut import (
    brute_force_maxcut,
    cut_value,
    draw_cut_solution,
    draw_graph,
    index_to_bitstring,
    make_graph,
)
from walsh.quantum_sub import draw_subcircuit, run_subcircuit


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # grafo
    "N_VERTICES": 4,
    "PROB_ARESTA": 1.0,
    "COM_PESO": False,
    "PESO_MIN": 1,
    "PESO_MAX": 3,
    "GRAPH_SEED": 42,
    "REVERSE_BITS": False,

    # loader Walsh
    "ER": 1.0,
    "MAX_TERMS": None,
    "A_MAX": math.pi / 4,
    "INIT_BETA": 0.35,

    # treino local de cada j
    "MAX_EPOCHS": 100,
    "LR": 0.20,
    "FD_EPS": 1e-2,
    "LOSS_TOL_LAST5": 1e-3,
    "PATIENCE_LAST5": 5,

    # poda / saturação
    "COEF_ZERO_TOL": 1e-3,
    "IMPROVEMENT_TOL": 1e-3,
    "SATURATION_PATIENCE": 100,

    # estratégia de varredura dos índices
    # opções: "ascending", "descending", "random", "middle_out"
    "INDEX_STRATEGY": "random",
    "INDEX_RANDOM_SEED": 42,
    "ENABLE_BACKWARD_PRUNING": False,
    "BACKWARD_PASSES": 1,
    "PRUNE_IMPROVEMENT_TOL": 1e-6,

    # saída
    "OUTDIR": "results_maxcut_walsh_seq",
    "SHOW_PLOTS": True,
}


# ============================================================
# ESTRUTURAS
# ============================================================
@dataclass
class StepResult:
    j: int
    kept: bool
    beta: float
    coefficient: float
    loss: float
    expected_cut: float
    best_measured_cut: float
    success_probability_ancilla_1: float
    epochs_run: int
    stop_reason: str
    history_path: str


# ============================================================
# UTIL
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bits_to_str(bits: List[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def bounded_coef(beta: float, a_max: float) -> float:
    return a_max * math.tanh(beta)


def build_avec_from_dict(n: int, coeffs: Dict[int, float]) -> np.ndarray:
    avec = np.zeros(2**n, dtype=float)
    for j, val in coeffs.items():
        if 0 <= j < 2**n:
            avec[j] = float(val)
    return avec


def get_all_cut_values(G: nx.Graph, reverse_bits: bool) -> np.ndarray:
    n = G.number_of_nodes()
    vals = np.zeros(2**n, dtype=float)
    for idx in range(2**n):
        bits = index_to_bitstring(idx, n, reverse_bits=reverse_bits)
        vals[idx] = float(cut_value(bits, G))
    return vals


def best_classical_solution(
    G: nx.Graph, reverse_bits: bool
) -> Tuple[float, List[int], int, str]:
    best_cut, best_bits, best_idx = brute_force_maxcut(G, reverse_bits=reverse_bits)
    return float(best_cut), best_bits, int(best_idx), bits_to_str(best_bits)


def run_current_circuit(
    G: nx.Graph,
    n: int,
    coeffs: Dict[int, float],
    reverse_bits: bool,
    er: float,
    cut_vals: np.ndarray,
) -> Dict[str, object]:
    avec = build_avec_from_dict(n, coeffs)
    active_js = [j for j, v in sorted(coeffs.items()) if j != 0 and abs(v) > 0.0]

    out = run_subcircuit(
        avec=avec,
        n=n,
        list_j=active_js,
        er=er,
        return_full_state=False,
    )

    success_p = float(out["success_probability_ancilla_1"])
    probs = np.asarray(out["postselected_probabilities"], dtype=float)

    if (success_p < 1e-14) or (not np.all(np.isfinite(probs))) or (float(probs.sum()) < 1e-14):
        probs = np.zeros(2**n, dtype=float)
        expected_cut = 0.0
        best_idx = 0
        best_cut = 0.0
    else:
        probs = probs / probs.sum()
        expected_cut = float(np.sum(probs * cut_vals))
        best_idx = int(np.argmax(probs))
        best_cut = float(cut_vals[best_idx])

    return {
        "success_probability_ancilla_1": success_p,
        "probs": probs,
        "expected_cut": expected_cut,
        "best_measured_idx": best_idx,
        "best_measured_cut": best_cut,
    }


def objective_for_single_j(
    beta: float,
    j: int,
    frozen_coeffs: Dict[int, float],
    G: nx.Graph,
    n: int,
    reverse_bits: bool,
    er: float,
    a_max: float,
    cut_vals: np.ndarray,
) -> Tuple[float, Dict[str, object]]:
    coeffs = dict(frozen_coeffs)
    coeffs[j] = bounded_coef(beta, a_max)

    metrics = run_current_circuit(
        G=G,
        n=n,
        coeffs=coeffs,
        reverse_bits=reverse_bits,
        er=er,
        cut_vals=cut_vals,
    )

    loss = - float(metrics["expected_cut"])
    return loss, metrics


def make_candidate_js(n: int, cfg: Dict[str, object]) -> List[int]:
    max_terms = cfg["MAX_TERMS"]
    if max_terms is None:
        base = list(range(1, 2**n))
    else:
        base = list(range(1, min(2**n, int(max_terms) + 1)))

    strategy = str(cfg.get("INDEX_STRATEGY", "ascending")).lower()

    if strategy == "ascending":
        return base

    if strategy == "descending":
        return list(reversed(base))

    if strategy == "random":
        rng = random.Random(int(cfg.get("INDEX_RANDOM_SEED", 123)))
        out = base[:]
        rng.shuffle(out)
        return out

    if strategy == "middle_out":
        m = len(base)
        if m == 0:
            return []
        left = (m - 1) // 2
        right = left + 1
        order = [base[left]]
        step = 1
        while len(order) < m:
            li = left - step
            ri = left + step
            if li >= 0:
                order.append(base[li])
            if ri < m:
                order.append(base[ri])
            step += 1
        return order[:m]

    raise ValueError(f"INDEX_STRATEGY desconhecida: {strategy}")


# ============================================================
# TREINO DE UM TERMO j
# ============================================================
def train_one_unitary(
    j: int,
    frozen_coeffs: Dict[int, float],
    G: nx.Graph,
    n: int,
    reverse_bits: bool,
    cfg: Dict[str, object],
    outdir: Path,
    cut_vals: np.ndarray,
) -> Tuple[StepResult, Dict[int, float]]:
    base_beta = float(cfg["INIT_BETA"])
    beta = base_beta if (j % 2 == 0) else -base_beta

    lr = float(cfg["LR"])
    fd_eps = float(cfg["FD_EPS"])
    max_epochs = int(cfg["MAX_EPOCHS"])
    loss_tol_last5 = float(cfg["LOSS_TOL_LAST5"])
    coef_zero_tol = float(cfg["COEF_ZERO_TOL"])
    er = float(cfg["ER"])
    a_max = float(cfg["A_MAX"])

    history = []
    best_beta = beta
    best_loss = float("inf")
    stop_reason = "max_epochs"

    for epoch in range(max_epochs):
        loss0, _ = objective_for_single_j(
            beta, j, frozen_coeffs, G, n, reverse_bits, er, a_max, cut_vals
        )
        loss_plus, _ = objective_for_single_j(
            beta + fd_eps, j, frozen_coeffs, G, n, reverse_bits, er, a_max, cut_vals
        )
        loss_minus, _ = objective_for_single_j(
            beta - fd_eps, j, frozen_coeffs, G, n, reverse_bits, er, a_max, cut_vals
        )

        grad = (loss_plus - loss_minus) / (2.0 * fd_eps)
        beta = beta - lr * grad

        loss1, metrics1 = objective_for_single_j(
            beta, j, frozen_coeffs, G, n, reverse_bits, er, a_max, cut_vals
        )
        coef = bounded_coef(beta, a_max)

        history.append({
            "epoch": epoch,
            "beta": float(beta),
            "coefficient": float(coef),
            "loss": float(loss1),
            "expected_cut": float(metrics1["expected_cut"]),
            "best_measured_cut": float(metrics1["best_measured_cut"]),
            "success_probability_ancilla_1": float(metrics1["success_probability_ancilla_1"]),
            "grad": float(grad),
        })

        if loss1 < best_loss:
            best_loss = loss1
            best_beta = beta

        if len(history) >= int(cfg["PATIENCE_LAST5"]):
            last_losses = [row["loss"] for row in history[-int(cfg["PATIENCE_LAST5"]):]]
            if max(last_losses) - min(last_losses) < loss_tol_last5:
                stop_reason = "loss_plateau_last5"
                break

    final_coef = bounded_coef(best_beta, a_max)
    kept = abs(final_coef) >= coef_zero_tol

    new_coeffs = dict(frozen_coeffs)
    if kept:
        new_coeffs[j] = float(final_coef)
    else:
        new_coeffs[j] = 0.0

    final_metrics = run_current_circuit(
        G=G,
        n=n,
        coeffs=new_coeffs,
        reverse_bits=reverse_bits,
        er=er,
        cut_vals=cut_vals,
    )

    hist_dir = outdir / "histories"
    ensure_dir(hist_dir)

    history_path = hist_dir / f"history_j{j:03d}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    step = StepResult(
        j=j,
        kept=kept,
        beta=float(best_beta),
        coefficient=float(final_coef if kept else 0.0),
        loss=float(-final_metrics["expected_cut"]),
        expected_cut=float(final_metrics["expected_cut"]),
        best_measured_cut=float(final_metrics["best_measured_cut"]),
        success_probability_ancilla_1=float(final_metrics["success_probability_ancilla_1"]),
        epochs_run=len(history),
        stop_reason=stop_reason if kept else stop_reason + "_and_discarded_near_zero",
        history_path=str(history_path),
    )
    return step, new_coeffs


def backward_pruning_pass(
    coeffs: Dict[int, float],
    G: nx.Graph,
    n: int,
    reverse_bits: bool,
    cfg: Dict[str, object],
    cut_vals: np.ndarray,
    cut_reference: float,
) -> Tuple[Dict[int, float], List[Dict[str, object]]]:
    er = float(cfg["ER"])
    tol = float(cfg.get("PRUNE_IMPROVEMENT_TOL", 1e-6))

    active_js = [j for j, v in sorted(coeffs.items(), reverse=True) if j != 0 and abs(v) > 0.0]
    pruned_log = []

    for j in active_js:
        test_coeffs = dict(coeffs)
        old_val = float(test_coeffs.get(j, 0.0))
        test_coeffs[j] = 0.0

        metrics = run_current_circuit(
            G=G,
            n=n,
            coeffs=test_coeffs,
            reverse_bits=reverse_bits,
            er=er,
            cut_vals=cut_vals,
        )
        new_cut = float(metrics["expected_cut"])
        delta = cut_reference - new_cut

        remove_it = delta <= tol
        if remove_it:
            coeffs = test_coeffs
            cut_reference = new_cut

        pruned_log.append({
            "j": int(j),
            "old_coefficient": old_val,
            "removed": bool(remove_it),
            "delta_expected_cut": float(delta),
            "expected_cut_after_test": float(new_cut),
        })

    return coeffs, pruned_log


# ============================================================
# PLOTS
# ============================================================
def save_global_plots(steps: List[StepResult], outdir: Path, show_plots: bool = False) -> None:
    plot_dir = outdir / "plots"
    ensure_dir(plot_dir)

    js = [s.j for s in steps]
    coeffs = [s.coefficient for s in steps]
    losses = [s.loss for s in steps]
    expecteds = [s.expected_cut for s in steps]
    bests = [s.best_measured_cut for s in steps]
    keeps = [1 if s.kept else 0 for s in steps]

    plt.figure(figsize=(8, 4))
    plt.plot(js, coeffs, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("j")
    plt.ylabel("coeficiente")
    plt.title("Coeficientes Walsh treinados")
    plt.tight_layout()
    plt.savefig(plot_dir / "trained_coefficients.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(js, losses, marker="o")
    plt.xlabel("ordem de visita")
    plt.ylabel("loss = - <H_cut>")
    plt.title("Loss por termo visitado")
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_progress.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(js, expecteds, marker="o", label="expected_cut")
    plt.plot(js, bests, marker="s", label="best_measured_cut")
    plt.xlabel("ordem de visita")
    plt.ylabel("cut")
    plt.title("Cut esperado e cut medido")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "cut_progress.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.step(js, keeps, where="mid")
    plt.yticks([0, 1], ["discard", "keep"])
    plt.xlabel("ordem de visita")
    plt.ylabel("status")
    plt.title("Termos mantidos ou descartados")
    plt.tight_layout()
    plt.savefig(plot_dir / "kept_vs_discarded.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()


# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run_sequential_walsh_maxcut(cfg: Dict[str, object]) -> Dict[str, object]:
    t0 = time.time()

    outdir = Path(cfg["OUTDIR"])
    ensure_dir(outdir)

    G = make_graph(
        n_vertices=int(cfg["N_VERTICES"]),
        prob_aresta=float(cfg["PROB_ARESTA"]),
        com_peso=bool(cfg["COM_PESO"]),
        peso_min=int(cfg["PESO_MIN"]),
        peso_max=int(cfg["PESO_MAX"]),
        seed=int(cfg["GRAPH_SEED"]),
    )
    n = G.number_of_nodes()
    reverse_bits = bool(cfg["REVERSE_BITS"])

    cut_vals = get_all_cut_values(G, reverse_bits=reverse_bits)

    plt.ioff()
    draw_graph(G, title="Grafo MaxCut")
    plt.savefig(outdir / "graph.png", dpi=180)
    plt.close("all")

    best_cut, best_bits, best_idx, best_bitstring = best_classical_solution(G, reverse_bits=reverse_bits)
    draw_cut_solution(G, best_bits, title=f"Solução ótima clássica = {best_cut:.3f}")
    plt.savefig(outdir / "one_optimal_cut_solution.png", dpi=180)
    plt.close("all")

    coeffs: Dict[int, float] = {0: 0.0}
    steps: List[StepResult] = []

    candidate_js = make_candidate_js(n, cfg)

    no_improve_counter = 0
    prev_expected_cut = 0.0

    for visit_idx, j in enumerate(candidate_js, start=1):
        step, coeffs = train_one_unitary(
            j=j,
            frozen_coeffs=coeffs,
            G=G,
            n=n,
            reverse_bits=reverse_bits,
            cfg=cfg,
            outdir=outdir,
            cut_vals=cut_vals,
        )
        steps.append(step)

        improvement = step.expected_cut - prev_expected_cut
        if improvement < float(cfg["IMPROVEMENT_TOL"]):
            no_improve_counter += 1
        else:
            no_improve_counter = 0
        prev_expected_cut = step.expected_cut

        print(
            f"visit={visit_idx:3d} | j={j:3d} | kept={step.kept} | coef={step.coefficient:+.6f} | "
            f"loss={step.loss:.6f} | E[cut]={step.expected_cut:.6f} | "
            f"best_measured_cut={step.best_measured_cut:.6f}"
        )

        if no_improve_counter >= int(cfg["SATURATION_PATIENCE"]):
            print("Saturou: parando adição de novos termos.")
            break

    pruning_log = []
    if bool(cfg.get("ENABLE_BACKWARD_PRUNING", False)):
        current_metrics = run_current_circuit(
            G=G,
            n=n,
            coeffs=coeffs,
            reverse_bits=reverse_bits,
            er=float(cfg["ER"]),
            cut_vals=cut_vals,
        )
        current_cut = float(current_metrics["expected_cut"])

        for _ in range(int(cfg.get("BACKWARD_PASSES", 1))):
            coeffs, one_pass = backward_pruning_pass(
                coeffs=coeffs,
                G=G,
                n=n,
                reverse_bits=reverse_bits,
                cfg=cfg,
                cut_vals=cut_vals,
                cut_reference=current_cut,
            )
            pruning_log.extend(one_pass)
            current_metrics = run_current_circuit(
                G=G,
                n=n,
                coeffs=coeffs,
                reverse_bits=reverse_bits,
                er=float(cfg["ER"]),
                cut_vals=cut_vals,
            )
            current_cut = float(current_metrics["expected_cut"])

    final_metrics = run_current_circuit(
        G=G,
        n=n,
        coeffs=coeffs,
        reverse_bits=reverse_bits,
        er=float(cfg["ER"]),
        cut_vals=cut_vals,
    )

    final_probs = np.asarray(final_metrics["probs"], dtype=float)
    final_best_idx = int(np.argmax(final_probs)) if final_probs.sum() > 0 else 0
    final_best_bits = index_to_bitstring(final_best_idx, n, reverse_bits=reverse_bits)
    final_best_cut = float(cut_vals[final_best_idx])

    draw_cut_solution(
        G,
        final_best_bits,
        title=f"Solução final medida pelo circuito = {final_best_cut:.3f}"
    )
    plt.savefig(outdir / "final_measured_cut_solution.png", dpi=180)
    plt.close("all")

    circuit_dir = outdir / "circuits"
    ensure_dir(circuit_dir)

    final_avec = build_avec_from_dict(n, coeffs)
    final_js = [idx for idx, val in sorted(coeffs.items()) if idx != 0 and abs(val) > 0.0]
    final_txt = draw_subcircuit(final_avec, n, final_js, er=float(cfg["ER"]))

    with open(circuit_dir / "final_circuit.txt", "w", encoding="utf-8") as f:
        f.write(str(final_txt))

    save_global_plots(steps, outdir, show_plots=bool(cfg["SHOW_PLOTS"]))

    summary = {
        "config": cfg,
        "n_qubits_register": n,
        "best_cut_bruteforce": float(best_cut),
        "best_bitstring_bruteforce": best_bitstring,
        "candidate_order": candidate_js,
        "trained_coefficients": {str(k): float(v) for k, v in sorted(coeffs.items())},
        "active_indices": [int(k) for k, v in sorted(coeffs.items()) if k != 0 and abs(v) > 0.0],
        "final_loss": float(-final_metrics["expected_cut"]),
        "final_expected_cut": float(final_metrics["expected_cut"]),
        "final_best_measured_index": int(final_best_idx),
        "final_best_measured_bitstring": bits_to_str(final_best_bits),
        "final_best_measured_cut": float(final_best_cut),
        "final_success_probability_ancilla_1": float(final_metrics["success_probability_ancilla_1"]),
        "n_steps": len(steps),
        "elapsed_seconds": time.time() - t0,
        "steps": [asdict(s) for s in steps],
        "pruning_log": pruning_log,
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_sequential_walsh_maxcut(CONFIG)

    print("\nResumo final")
    print(json.dumps({
        "best_cut_bruteforce": summary["best_cut_bruteforce"],
        "final_loss": summary["final_loss"],
        "final_expected_cut": summary["final_expected_cut"],
        "final_best_measured_bitstring": summary["final_best_measured_bitstring"],
        "final_best_measured_cut": summary["final_best_measured_cut"],
        "active_indices": summary["active_indices"],
    }, indent=2))
