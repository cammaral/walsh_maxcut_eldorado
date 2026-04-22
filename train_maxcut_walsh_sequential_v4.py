from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from walsh.quantum_sub import draw_subcircuit, run_single_unitary, run_subcircuit


CONFIG = {
    # grafo
    "N_VERTICES": 4,
    "PROB_ARESTA": 0.7,
    "COM_PESO": True,
    "PESO_MIN": 1,
    "PESO_MAX": 3,
    "GRAPH_SEED": 42,
    "REVERSE_BITS": False,

    # loader Walsh
    "ER": 1.0,
    "MAX_TERMS": None,            # None -> tenta todos os j=0,...,2^n-1
    "A_MAX": math.pi / 4,         # a_j = A_MAX * tanh(beta)
    "INIT_BETA": 0.35,

    # treino local de cada j
    "MAX_EPOCHS": 100,
    "LR": 0.20,
    "FD_EPS": 1e-2,
    "LOSS_TOL_LAST5": 1e-5,
    "PATIENCE_LAST5": 5,

    # poda / parada
    "COEF_ZERO_TOL": 1e-3,
    "SATURATION_PATIENCE": 6,
    "PROB_SUPPORT_TOL": 1e-14,

    # saída
    "OUTDIR": "results_maxcut_walsh_seq_v4",
    "SHOW_PLOTS": False,
}


@dataclass
class StepResult:
    j: int
    kept: bool
    beta: float
    coefficient_candidate: float
    coefficient_kept: float
    local_loss: float
    local_expected_cut: float
    local_most_probable_cut: float
    local_most_probable_bitstring: str
    local_best_found_cut: float
    local_best_found_bitstring: str
    local_success_probability_ancilla_1: float
    global_loss_after_step: float
    global_expected_cut_after_step: float
    global_most_probable_cut_after_step: float
    global_most_probable_bitstring_after_step: str
    global_best_found_cut_after_step: float
    global_best_found_bitstring_after_step: str
    global_success_probability_ancilla_1_after_step: float
    epochs_run: int
    stop_reason: str
    history_path: str
    plot_path: str


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


def extract_conditional_probs_from_full_state(full_state: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    state = np.asarray(full_state)
    state = state.reshape(2, 2**n)
    branch = state[1]
    raw_probs = np.abs(branch) ** 2
    success_prob = float(np.real(raw_probs.sum()))

    if (not np.isfinite(success_prob)) or success_prob < 1e-18:
        return np.zeros(2**n, dtype=float), 0.0

    cond_probs = raw_probs / success_prob
    cond_probs = np.asarray(cond_probs, dtype=float)
    return cond_probs, success_prob


def summarize_distribution(
    probs: np.ndarray,
    cut_vals: np.ndarray,
    n: int,
    reverse_bits: bool,
    support_tol: float,
) -> Dict[str, object]:
    probs = np.asarray(probs, dtype=float)

    if (probs.size == 0) or (not np.all(np.isfinite(probs))) or float(probs.sum()) < 1e-18:
        most_idx = 0
        best_idx = 0
        expected_cut = 0.0
        probs = np.zeros_like(cut_vals, dtype=float)
    else:
        probs = probs / probs.sum()
        expected_cut = float(np.sum(probs * cut_vals))
        most_idx = int(np.argmax(probs))
        support = np.where(probs > support_tol)[0]
        if len(support) == 0:
            best_idx = most_idx
        else:
            best_idx = int(support[np.argmax(cut_vals[support])])

    most_bits = index_to_bitstring(most_idx, n, reverse_bits=reverse_bits)
    best_bits = index_to_bitstring(best_idx, n, reverse_bits=reverse_bits)

    return {
        "probs": probs,
        "expected_cut": expected_cut,
        "most_probable_idx": most_idx,
        "most_probable_bitstring": bits_to_str(most_bits),
        "most_probable_cut": float(cut_vals[most_idx]),
        "best_found_idx": best_idx,
        "best_found_bitstring": bits_to_str(best_bits),
        "best_found_cut": float(cut_vals[best_idx]),
    }


def run_current_global_circuit(
    n: int,
    coeffs: Dict[int, float],
    reverse_bits: bool,
    er: float,
    cut_vals: np.ndarray,
    support_tol: float,
) -> Dict[str, object]:
    avec = build_avec_from_dict(n, coeffs)
    active_js = [j for j, v in sorted(coeffs.items()) if abs(v) > 0.0]

    out = run_subcircuit(
        avec=avec,
        n=n,
        list_j=active_js,
        er=er,
        return_full_state=True,
    )

    probs, success_p = extract_conditional_probs_from_full_state(out["full_state"], n)
    summary = summarize_distribution(probs, cut_vals, n, reverse_bits, support_tol)
    summary["success_probability_ancilla_1"] = success_p
    return summary


def objective_for_single_j_local(
    beta: float,
    j: int,
    n: int,
    reverse_bits: bool,
    er: float,
    a_max: float,
    cut_vals: np.ndarray,
    support_tol: float,
) -> Tuple[float, Dict[str, object]]:
    avec = np.zeros(2**n, dtype=float)
    avec[j] = bounded_coef(beta, a_max)

    out = run_single_unitary(
        avec=avec,
        n=n,
        j=j,
        er=er,
        return_full_state=True,
    )

    probs, success_p = extract_conditional_probs_from_full_state(out["full_state"], n)
    summary = summarize_distribution(probs, cut_vals, n, reverse_bits, support_tol)
    summary["success_probability_ancilla_1"] = success_p

    loss = -float(summary["expected_cut"])
    return loss, summary


def save_step_history_plot(history: List[Dict[str, float]], j: int, plot_path: Path, show_plots: bool = False) -> None:
    if len(history) == 0:
        return

    epochs = [row["epoch"] for row in history]
    losses = [row["local_loss"] for row in history]
    coefs = [row["coefficient"] for row in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, losses, marker="o", label="local loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("local loss = - expected_cut(single U_j)")

    ax2 = ax1.twinx()
    ax2.plot(epochs, coefs, marker="s", linestyle="--", label="coefficient")
    ax2.set_ylabel("coefficient a_j")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"Treino local do termo j={j}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(fig)


def train_one_unitary_local_then_eval_global(
    j: int,
    frozen_coeffs: Dict[int, float],
    n: int,
    reverse_bits: bool,
    cfg: Dict[str, object],
    outdir: Path,
    cut_vals: np.ndarray,
) -> Tuple[StepResult, Dict[int, float], float, Dict[str, object]]:
    base_beta = float(cfg["INIT_BETA"])
    beta = base_beta if (j % 2 == 0) else -base_beta

    lr = float(cfg["LR"])
    fd_eps = float(cfg["FD_EPS"])
    max_epochs = int(cfg["MAX_EPOCHS"])
    loss_tol_last5 = float(cfg["LOSS_TOL_LAST5"])
    coef_zero_tol = float(cfg["COEF_ZERO_TOL"])
    er = float(cfg["ER"])
    a_max = float(cfg["A_MAX"])
    support_tol = float(cfg["PROB_SUPPORT_TOL"])

    history: List[Dict[str, float]] = []
    best_beta = beta
    best_loss = float("inf")
    best_metrics: Optional[Dict[str, object]] = None
    stop_reason = "max_epochs"

    for epoch in range(max_epochs):
        loss_plus, _ = objective_for_single_j_local(
            beta + fd_eps, j, n, reverse_bits, er, a_max, cut_vals, support_tol
        )
        loss_minus, _ = objective_for_single_j_local(
            beta - fd_eps, j, n, reverse_bits, er, a_max, cut_vals, support_tol
        )
        grad = (loss_plus - loss_minus) / (2.0 * fd_eps)
        beta = beta - lr * grad

        loss1, metrics1 = objective_for_single_j_local(
            beta, j, n, reverse_bits, er, a_max, cut_vals, support_tol
        )
        coef = bounded_coef(beta, a_max)

        history.append({
            "epoch": epoch,
            "beta": float(beta),
            "coefficient": float(coef),
            "local_loss": float(loss1),
            "local_expected_cut": float(metrics1["expected_cut"]),
            "local_most_probable_cut": float(metrics1["most_probable_cut"]),
            "local_most_probable_bitstring": str(metrics1["most_probable_bitstring"]),
            "local_best_found_cut": float(metrics1["best_found_cut"]),
            "local_best_found_bitstring": str(metrics1["best_found_bitstring"]),
            "local_success_probability_ancilla_1": float(metrics1["success_probability_ancilla_1"]),
            "grad": float(grad),
        })

        if loss1 < best_loss:
            best_loss = loss1
            best_beta = beta
            best_metrics = metrics1

        if len(history) >= int(cfg["PATIENCE_LAST5"]):
            last_losses = [row["local_loss"] for row in history[-int(cfg["PATIENCE_LAST5"]):]]
            if max(last_losses) - min(last_losses) < loss_tol_last5:
                stop_reason = "local_loss_plateau_last5"
                break

    if best_metrics is None:
        best_loss, best_metrics = objective_for_single_j_local(
            best_beta, j, n, reverse_bits, er, a_max, cut_vals, support_tol
        )

    coefficient_candidate = bounded_coef(best_beta, a_max)
    kept = abs(coefficient_candidate) >= coef_zero_tol

    updated_coeffs = dict(frozen_coeffs)
    if kept:
        updated_coeffs[j] = float(coefficient_candidate)
    else:
        updated_coeffs[j] = 0.0
        stop_reason += "_discarded_near_zero"

    global_metrics_after_step = run_current_global_circuit(
        n=n,
        coeffs=updated_coeffs,
        reverse_bits=reverse_bits,
        er=er,
        cut_vals=cut_vals,
        support_tol=support_tol,
    )

    hist_dir = outdir / "histories"
    plot_dir = outdir / "step_plots"
    ensure_dir(hist_dir)
    ensure_dir(plot_dir)

    history_path = hist_dir / f"history_j{j:03d}.json"
    plot_path = plot_dir / f"history_j{j:03d}.png"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_step_history_plot(history, j, plot_path, show_plots=bool(cfg["SHOW_PLOTS"]))

    step = StepResult(
        j=j,
        kept=kept,
        beta=float(best_beta),
        coefficient_candidate=float(coefficient_candidate),
        coefficient_kept=float(coefficient_candidate if kept else 0.0),
        local_loss=float(best_loss),
        local_expected_cut=float(best_metrics["expected_cut"]),
        local_most_probable_cut=float(best_metrics["most_probable_cut"]),
        local_most_probable_bitstring=str(best_metrics["most_probable_bitstring"]),
        local_best_found_cut=float(best_metrics["best_found_cut"]),
        local_best_found_bitstring=str(best_metrics["best_found_bitstring"]),
        local_success_probability_ancilla_1=float(best_metrics["success_probability_ancilla_1"]),
        global_loss_after_step=float(-global_metrics_after_step["expected_cut"]),
        global_expected_cut_after_step=float(global_metrics_after_step["expected_cut"]),
        global_most_probable_cut_after_step=float(global_metrics_after_step["most_probable_cut"]),
        global_most_probable_bitstring_after_step=str(global_metrics_after_step["most_probable_bitstring"]),
        global_best_found_cut_after_step=float(global_metrics_after_step["best_found_cut"]),
        global_best_found_bitstring_after_step=str(global_metrics_after_step["best_found_bitstring"]),
        global_success_probability_ancilla_1_after_step=float(global_metrics_after_step["success_probability_ancilla_1"]),
        epochs_run=len(history),
        stop_reason=stop_reason,
        history_path=str(history_path),
        plot_path=str(plot_path),
    )
    return step, updated_coeffs, float(coefficient_candidate), global_metrics_after_step


def save_global_plots(
    steps: List[StepResult],
    coeffs_kept: Dict[int, float],
    coeffs_candidates: Dict[int, float],
    total_possible_params: int,
    outdir: Path,
    show_plots: bool = False,
) -> None:
    if len(steps) == 0:
        return

    plot_dir = outdir / "plots"
    ensure_dir(plot_dir)

    js = [s.j for s in steps]
    kept_coeffs = [s.coefficient_kept for s in steps]
    candidate_coeffs = [s.coefficient_candidate for s in steps]
    local_losses = [s.local_loss for s in steps]
    global_losses = [s.global_loss_after_step for s in steps]
    global_expecteds = [s.global_expected_cut_after_step for s in steps]
    global_most_prob_cuts = [s.global_most_probable_cut_after_step for s in steps]
    global_best_found_cuts = [s.global_best_found_cut_after_step for s in steps]
    keeps = [1 if s.kept else 0 for s in steps]

    all_js = list(range(total_possible_params))
    full_kept = [abs(float(coeffs_kept.get(j, 0.0))) for j in all_js]
    full_candidates = [abs(float(coeffs_candidates.get(j, 0.0))) for j in all_js]

    plt.figure(figsize=(9, 4))
    plt.bar(all_js, full_candidates)
    plt.xlabel("j")
    plt.ylabel("|coeficiente candidato|")
    plt.title("Magnitude dos coeficientes candidatos")
    plt.tight_layout()
    plt.savefig(plot_dir / "candidate_coefficients_magnitude.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(all_js, full_kept)
    plt.xlabel("j")
    plt.ylabel("|coeficiente mantido|")
    plt.title("Magnitude dos coeficientes mantidos")
    plt.tight_layout()
    plt.savefig(plot_dir / "kept_coefficients_magnitude.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(js, candidate_coeffs, marker="o", label="candidate")
    plt.plot(js, kept_coeffs, marker="s", label="kept")
    plt.xlabel("j")
    plt.ylabel("coeficiente")
    plt.title("Coeficiente candidato e mantido por passo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "coefficients_by_step.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(js, local_losses, marker="o", label="local subloss")
    plt.plot(js, global_losses, marker="s", label="global loss after step")
    plt.xlabel("j")
    plt.ylabel("loss")
    plt.title("Loss local e global por termo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_progress_local_global.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(js, global_expecteds, marker="o", label="expected cut (global)")
    plt.plot(js, global_most_prob_cuts, marker="s", label="cut da bitstring mais provável (global)")
    plt.plot(js, global_best_found_cuts, marker="^", label="melhor cut encontrado (global)")
    plt.xlabel("j")
    plt.ylabel("cut")
    plt.title("Métricas globais por passo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "cut_progress_global.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.step(js, keeps, where="mid")
    plt.yticks([0, 1], ["discard", "keep"])
    plt.xlabel("j")
    plt.ylabel("status")
    plt.title("Termos mantidos ou descartados")
    plt.tight_layout()
    plt.savefig(plot_dir / "kept_vs_discarded.png", dpi=180)
    if show_plots:
        plt.show()
    plt.close()


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
    total_possible_params = 2**n

    cut_vals = get_all_cut_values(G, reverse_bits=reverse_bits)

    plt.ioff()
    draw_graph(G, title="Grafo MaxCut")
    plt.savefig(outdir / "graph.png", dpi=180)
    plt.close("all")

    best_cut, best_bits, _, best_bitstring = best_classical_solution(G, reverse_bits=reverse_bits)
    draw_cut_solution(G, best_bits, title=f"Solução ótima clássica = {best_cut:.3f}")
    plt.savefig(outdir / "one_optimal_cut_solution.png", dpi=180)
    plt.close("all")

    coeffs_kept: Dict[int, float] = {}
    coeffs_candidates: Dict[int, float] = {}
    steps: List[StepResult] = []

    max_terms = cfg["MAX_TERMS"]
    if max_terms is None:
        candidate_js = list(range(total_possible_params))
    else:
        candidate_js = list(range(0, min(total_possible_params, int(max_terms))))

    no_keep_counter = 0

    for j in candidate_js:
        step, coeffs_kept, candidate_coef, global_metrics = train_one_unitary_local_then_eval_global(
            j=j,
            frozen_coeffs=coeffs_kept,
            n=n,
            reverse_bits=reverse_bits,
            cfg=cfg,
            outdir=outdir,
            cut_vals=cut_vals,
        )
        coeffs_candidates[j] = float(candidate_coef)
        steps.append(step)

        if step.kept:
            no_keep_counter = 0
        else:
            no_keep_counter += 1

        print(
            f"j={j:3d} | kept={step.kept} | candidate={step.coefficient_candidate:+.6f} | "
            f"kept_coef={step.coefficient_kept:+.6f} | local_loss={step.local_loss:.6f} | "
            f"global_E[cut]={step.global_expected_cut_after_step:.6f} | "
            f"global_most_prob_cut={step.global_most_probable_cut_after_step:.6f}"
        )

        if no_keep_counter >= int(cfg["SATURATION_PATIENCE"]):
            print("Muitos coeficientes nulos seguidos: parando adição de novos termos.")
            break

    final_metrics = run_current_global_circuit(
        n=n,
        coeffs=coeffs_kept,
        reverse_bits=reverse_bits,
        er=float(cfg["ER"]),
        cut_vals=cut_vals,
        support_tol=float(cfg["PROB_SUPPORT_TOL"]),
    )

    final_most_prob_idx = int(final_metrics["most_probable_idx"])
    final_most_prob_bits = index_to_bitstring(final_most_prob_idx, n, reverse_bits=reverse_bits)
    draw_cut_solution(
        G,
        final_most_prob_bits,
        title=f"Bitstring global mais provável = {final_metrics['most_probable_cut']:.3f}",
    )
    plt.savefig(outdir / "final_global_most_probable_solution.png", dpi=180)
    plt.close("all")

    final_best_found_idx = int(final_metrics["best_found_idx"])
    final_best_found_bits = index_to_bitstring(final_best_found_idx, n, reverse_bits=reverse_bits)
    draw_cut_solution(
        G,
        final_best_found_bits,
        title=f"Melhor bitstring encontrada no global = {final_metrics['best_found_cut']:.3f}",
    )
    plt.savefig(outdir / "final_global_best_found_solution.png", dpi=180)
    plt.close("all")

    circuit_dir = outdir / "circuits"
    ensure_dir(circuit_dir)

    final_avec = build_avec_from_dict(n, coeffs_kept)
    final_js = [idx for idx, val in sorted(coeffs_kept.items()) if abs(val) > 0.0]
    final_fig = draw_subcircuit(final_avec, n, final_js, er=float(cfg["ER"]))
    if isinstance(final_fig, tuple):
        fig = final_fig[0]
    else:
        fig = final_fig
    try:
        fig.savefig(circuit_dir / "final_circuit.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    with open(circuit_dir / "final_active_indices.json", "w", encoding="utf-8") as f:
        json.dump({"active_indices": final_js}, f, indent=2)

    save_global_plots(
        steps=steps,
        coeffs_kept=coeffs_kept,
        coeffs_candidates=coeffs_candidates,
        total_possible_params=total_possible_params,
        outdir=outdir,
        show_plots=bool(cfg["SHOW_PLOTS"]),
    )

    n_kept = sum(1 for _, v in coeffs_kept.items() if abs(v) > 0.0)
    kept_fraction = float(n_kept / total_possible_params) if total_possible_params > 0 else 0.0

    summary = {
        "config": cfg,
        "n_qubits_register": n,
        "total_possible_parameters": int(total_possible_params),
        "total_attempted_parameters": int(len(candidate_js)),
        "total_kept_parameters": int(n_kept),
        "kept_fraction": kept_fraction,
        "best_cut_bruteforce": float(best_cut),
        "best_bitstring_bruteforce": best_bitstring,
        "trained_candidate_coefficients": {str(k): float(v) for k, v in sorted(coeffs_candidates.items())},
        "trained_kept_coefficients": {str(k): float(v) for k, v in sorted(coeffs_kept.items())},
        "active_indices": [int(k) for k, v in sorted(coeffs_kept.items()) if abs(v) > 0.0],
        "final_global_loss": float(-final_metrics["expected_cut"]),
        "final_global_expected_cut": float(final_metrics["expected_cut"]),
        "final_global_most_probable_index": int(final_metrics["most_probable_idx"]),
        "final_global_most_probable_bitstring": str(final_metrics["most_probable_bitstring"]),
        "final_global_most_probable_cut": float(final_metrics["most_probable_cut"]),
        "final_global_best_found_index": int(final_metrics["best_found_idx"]),
        "final_global_best_found_bitstring": str(final_metrics["best_found_bitstring"]),
        "final_global_best_found_cut": float(final_metrics["best_found_cut"]),
        "final_global_success_probability_ancilla_1": float(final_metrics["success_probability_ancilla_1"]),
        "n_steps": len(steps),
        "elapsed_seconds": time.time() - t0,
        "steps": [asdict(s) for s in steps],
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_sequential_walsh_maxcut(CONFIG)

    print("\nResumo final")
    print(json.dumps({
        "total_possible_parameters": summary["total_possible_parameters"],
        "total_kept_parameters": summary["total_kept_parameters"],
        "kept_fraction": summary["kept_fraction"],
        "best_cut_bruteforce": summary["best_cut_bruteforce"],
        "final_global_loss": summary["final_global_loss"],
        "final_global_expected_cut": summary["final_global_expected_cut"],
        "final_global_most_probable_bitstring": summary["final_global_most_probable_bitstring"],
        "final_global_most_probable_cut": summary["final_global_most_probable_cut"],
        "final_global_best_found_bitstring": summary["final_global_best_found_bitstring"],
        "final_global_best_found_cut": summary["final_global_best_found_cut"],
        "active_indices": summary["active_indices"],
    }, indent=2))
