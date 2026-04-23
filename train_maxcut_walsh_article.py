from __future__ import annotations

import csv
import hashlib
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
import pennylane as qml
import pennylane.numpy as pnp

from maxcut.maxcut import (
    brute_force_maxcut,
    cut_value,
    draw_cut_solution,
    draw_graph,
    index_to_bitstring,
    make_graph,
)
from walsh.quantum_sub import draw_subcircuit, run_subcircuit, run_subcircuit_differentiable


CONFIG = {
    # grafo
    "N_VERTICES": 4,
    "PROB_ARESTA": 1.0,
    "COM_PESO": False,
    "PESO_MIN": 1,
    "PESO_MAX": 3,
    "GRAPH_SEED": 42,
    "GRAPH_EDGES": None,
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
    "TRAIN_SEED": 0,
    "INIT_BETA_JITTER": 0.05,
    "OPTIMIZER": "adam",
    "ADAM_BETA1": 0.9,
    "ADAM_BETA2": 0.999,
    "ADAM_EPS": 1e-8,

    # poda / saturação
    "COEF_ZERO_TOL": 1e-3,
    "IMPROVEMENT_TOL": 1e-3,
    "SATURATION_PATIENCE": 100,

    # estratégia de varredura dos índices
    "INDEX_STRATEGY": "ascending",
    "INDEX_RANDOM_SEED": 42,
    "ENABLE_BACKWARD_PRUNING": False,
    "BACKWARD_PASSES": 1,
    "PRUNE_IMPROVEMENT_TOL": 1e-6,

    # saída
    "OUTDIR": "results_maxcut_walsh_article",
    "SHOW_PLOTS": False,
}


@dataclass
class StepResult:
    visit_idx: int
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
    improvement_vs_previous: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bits_to_str(bits: List[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def bounded_coef(beta: float, a_max: float) -> float:
    return a_max * math.tanh(beta)


def bounded_coef_pl(beta, a_max: float):
    return a_max * qml.math.tanh(beta)


def build_avec_from_dict(n: int, coeffs: Dict[int, float]) -> np.ndarray:
    avec = np.zeros(2**n, dtype=float)
    for j, val in coeffs.items():
        if 0 <= j < 2**n:
            avec[j] = float(val)
    return avec


def build_avec_from_dict_pl(n: int, coeffs: Dict[int, float]) -> pnp.ndarray:
    avec = pnp.zeros(2**n, dtype=float)
    for j, val in coeffs.items():
        if 0 <= j < 2**n:
            avec[j] = float(val)
    return avec


def build_avec_for_beta_pl(n: int, frozen_coeffs: Dict[int, float], j: int, beta, a_max: float) -> pnp.ndarray:
    avec = build_avec_from_dict_pl(n, frozen_coeffs)
    coef_j = bounded_coef_pl(beta, a_max)
    one_hot = pnp.zeros(2**n, dtype=float)
    one_hot[j] = 1.0
    return avec + coef_j * one_hot


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


def graph_to_serializable_edges(G: nx.Graph) -> List[List[float]]:
    out = []
    for u, v, data in sorted(G.edges(data=True), key=lambda t: (min(t[0], t[1]), max(t[0], t[1]))):
        w = float(data.get("weight", 1.0))
        out.append([int(min(u, v)), int(max(u, v)), w])
    return out


def graph_signature_from_edges(edges: List[List[float]]) -> str:
    norm = [(int(a), int(b), round(float(w), 12)) for a, b, w in edges]
    payload = json.dumps(norm, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_graph_from_cfg(cfg: Dict[str, object]) -> nx.Graph:
    explicit_edges = cfg.get("GRAPH_EDGES")
    n_vertices = int(cfg["N_VERTICES"])

    if explicit_edges is None:
        return make_graph(
            n_vertices=n_vertices,
            prob_aresta=float(cfg["PROB_ARESTA"]),
            com_peso=bool(cfg["COM_PESO"]),
            peso_min=int(cfg["PESO_MIN"]),
            peso_max=int(cfg["PESO_MAX"]),
            seed=int(cfg["GRAPH_SEED"]),
        )

    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    for u, v, w in explicit_edges:
        G.add_edge(int(u), int(v), weight=float(w))
    return G


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
        return_full_state=True,
    )

    success_p = float(out["success_probability_ancilla_1"])
    probs = np.asarray(out["postselected_probabilities"], dtype=float)
    full_state = np.asarray(out["full_state"], dtype=complex)
    postselected_state = np.asarray(out["postselected_state"], dtype=complex)

    if (success_p < 1e-14) or (not np.all(np.isfinite(probs))) or (float(probs.sum()) < 1e-14):
        probs = np.zeros(2**n, dtype=float)
        expected_cut = 0.0
        best_idx = 0
        best_cut = 0.0
        max_prob = 0.0
    else:
        probs = probs / probs.sum()
        expected_cut = float(np.sum(probs * cut_vals))
        best_idx = int(np.argmax(probs))
        best_cut = float(cut_vals[best_idx])
        max_prob = float(np.max(probs))

    return {
        "success_probability_ancilla_1": success_p,
        "probs": probs,
        "expected_cut": expected_cut,
        "best_measured_idx": best_idx,
        "best_measured_cut": best_cut,
        "max_postselected_probability": max_prob,
        "full_state": full_state,
        "postselected_state": postselected_state,
    }


def run_current_circuit_differentiable(
    G: nx.Graph,
    n: int,
    coeffs: Dict[int, float],
    reverse_bits: bool,
    er: float,
    cut_vals: np.ndarray,
) -> Dict[str, object]:
    del G, reverse_bits
    avec = build_avec_from_dict_pl(n, coeffs)
    active_js = [j for j, v in sorted(coeffs.items()) if j != 0 and abs(v) > 0.0]
    out = run_subcircuit_differentiable(
        avec=avec,
        n=n,
        list_j=active_js,
        er=er,
        return_full_state=True,
    )

    probs = out["postselected_probabilities"]
    cut_vals_pl = pnp.array(cut_vals, dtype=float)
    expected_cut = qml.math.sum(probs * cut_vals_pl)

    probs_np = np.asarray(probs, dtype=float)
    if probs_np.size == 0 or not np.all(np.isfinite(probs_np)):
        best_idx = 0
        best_cut = 0.0
        max_prob = 0.0
    else:
        best_idx = int(np.argmax(probs_np))
        best_cut = float(cut_vals[best_idx])
        max_prob = float(np.max(probs_np))

    return {
        "success_probability_ancilla_1": out["success_probability_ancilla_1"],
        "probs": probs,
        "expected_cut": expected_cut,
        "best_measured_idx": best_idx,
        "best_measured_cut": best_cut,
        "max_postselected_probability": max_prob,
        "full_state": out["full_state"],
        "postselected_state": out["postselected_state"],
    }


def objective_for_single_j(
    beta,
    j: int,
    frozen_coeffs: Dict[int, float],
    G: nx.Graph,
    n: int,
    reverse_bits: bool,
    er: float,
    a_max: float,
    cut_vals: np.ndarray,
) -> Tuple[object, Dict[str, object]]:
    coeffs = dict(frozen_coeffs)
    coeffs[j] = 0.0

    avec = build_avec_for_beta_pl(n, frozen_coeffs, j, beta, a_max)
    active_js = [idx for idx, val in sorted(coeffs.items()) if idx != 0 and abs(val) > 0.0]
    if j not in active_js:
        active_js.append(j)
        active_js = sorted(active_js)

    out = run_subcircuit_differentiable(
        avec=avec,
        n=n,
        list_j=active_js,
        er=er,
        return_full_state=True,
    )

    probs = out["postselected_probabilities"]
    cut_vals_pl = pnp.array(cut_vals, dtype=float)
    expected_cut = qml.math.sum(probs * cut_vals_pl)
    loss = -expected_cut

    metrics = {
        "success_probability_ancilla_1": out["success_probability_ancilla_1"],
        "probs": probs,
        "expected_cut": expected_cut,
        "full_state": out["full_state"],
        "postselected_state": out["postselected_state"],
    }
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
        center = (m - 1) // 2
        order = [base[center]]
        step = 1
        while len(order) < m:
            li = center - step
            ri = center + step
            if li >= 0:
                order.append(base[li])
            if ri < m:
                order.append(base[ri])
            step += 1
        return order[:m]

    raise ValueError(f"INDEX_STRATEGY desconhecida: {strategy}")


def initial_beta_for_j(j: int, cfg: Dict[str, object]) -> float:
    base_beta = float(cfg["INIT_BETA"])
    seed = int(cfg.get("TRAIN_SEED", 0))
    jitter = float(cfg.get("INIT_BETA_JITTER", 0.0))
    rng = random.Random((seed + 1) * 1000003 + j * 9176)
    signed_base = base_beta if (j % 2 == 0) else -base_beta
    return signed_base + rng.uniform(-jitter, jitter)


def save_history_json_and_csv(history: List[Dict[str, float]], history_path_json: Path) -> Path:
    with open(history_path_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    history_path_csv = history_path_json.with_suffix(".csv")
    if history:
        keys = list(history[0].keys())
        with open(history_path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    return history_path_csv


def train_one_unitary(
    j: int,
    visit_idx: int,
    frozen_coeffs: Dict[int, float],
    G: nx.Graph,
    n: int,
    reverse_bits: bool,
    cfg: Dict[str, object],
    outdir: Path,
    cut_vals: np.ndarray,
    previous_expected_cut: float,
) -> Tuple[StepResult, Dict[int, float]]:
    beta = pnp.array(initial_beta_for_j(j, cfg), requires_grad=True)

    lr = float(cfg["LR"])
    max_epochs = int(cfg["MAX_EPOCHS"])
    loss_tol_last5 = float(cfg["LOSS_TOL_LAST5"])
    coef_zero_tol = float(cfg["COEF_ZERO_TOL"])
    er = float(cfg["ER"])
    a_max = float(cfg["A_MAX"])

    optimizer_name = str(cfg.get("OPTIMIZER", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"OPTIMIZER desconhecido: {optimizer_name}. Esta versão usa Adam + backprop.")

    opt = qml.AdamOptimizer(
        stepsize=lr,
        beta1=float(cfg.get("ADAM_BETA1", 0.9)),
        beta2=float(cfg.get("ADAM_BETA2", 0.999)),
        eps=float(cfg.get("ADAM_EPS", 1e-8)),
    )

    history = []
    best_beta = float(beta)
    best_loss = float("inf")
    stop_reason = "max_epochs"

    def scalar_loss(beta_var):
        loss_val, _ = objective_for_single_j(
            beta_var, j, frozen_coeffs, G, n, reverse_bits, er, a_max, cut_vals
        )
        return loss_val

    grad_fn = qml.grad(scalar_loss)

    for epoch in range(max_epochs):
        grad_before = grad_fn(beta)
        beta, _ = opt.step_and_cost(scalar_loss, beta)

        loss1, metrics1 = objective_for_single_j(
            beta, j, frozen_coeffs, G, n, reverse_bits, er, a_max, cut_vals
        )
        loss1_float = float(loss1)
        coef = bounded_coef(float(beta), a_max)

        probs_np = np.asarray(metrics1["probs"], dtype=float)
        if probs_np.size == 0 or not np.all(np.isfinite(probs_np)):
            best_measured_cut_epoch = 0.0
            max_postselected_probability_epoch = 0.0
        else:
            best_idx_epoch = int(np.argmax(probs_np))
            best_measured_cut_epoch = float(cut_vals[best_idx_epoch])
            max_postselected_probability_epoch = float(np.max(probs_np))

        history.append({
            "epoch": epoch,
            "beta": float(beta),
            "coefficient": float(coef),
            "loss": loss1_float,
            "expected_cut": float(metrics1["expected_cut"]),
            "best_measured_cut": best_measured_cut_epoch,
            "success_probability_ancilla_1": float(metrics1["success_probability_ancilla_1"]),
            "max_postselected_probability": max_postselected_probability_epoch,
            "grad": float(grad_before),
        })

        if loss1_float < best_loss:
            best_loss = loss1_float
            best_beta = float(beta)

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

    history_path_json = hist_dir / f"history_visit{visit_idx:03d}_j{j:03d}.json"
    save_history_json_and_csv(history, history_path_json)

    expected_cut = float(final_metrics["expected_cut"])
    step = StepResult(
        visit_idx=int(visit_idx),
        j=j,
        kept=kept,
        beta=float(best_beta),
        coefficient=float(final_coef if kept else 0.0),
        loss=float(-expected_cut),
        expected_cut=expected_cut,
        best_measured_cut=float(final_metrics["best_measured_cut"]),
        success_probability_ancilla_1=float(final_metrics["success_probability_ancilla_1"]),
        epochs_run=len(history),
        stop_reason=stop_reason if kept else stop_reason + "_and_discarded_near_zero",
        history_path=str(history_path_json),
        improvement_vs_previous=float(expected_cut - previous_expected_cut),
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


def save_final_distributions_and_states(outdir: Path, final_metrics: Dict[str, object], cut_vals: np.ndarray) -> None:
    probs = np.asarray(final_metrics["probs"], dtype=float)
    full_state = np.asarray(final_metrics["full_state"])
    postselected_state = np.asarray(final_metrics["postselected_state"])

    np.save(outdir / "final_postselected_probabilities.npy", probs)
    np.save(outdir / "final_full_state_realimag.npy", np.column_stack([full_state.real, full_state.imag]))
    np.save(outdir / "final_postselected_state_realimag.npy", np.column_stack([postselected_state.real, postselected_state.imag]))

    topk = min(20, len(probs))
    top_indices = np.argsort(probs)[::-1][:topk]
    rows = []
    n = int(round(math.log2(len(probs)))) if len(probs) > 0 else 0
    for rank, idx in enumerate(top_indices, start=1):
        rows.append({
            "rank": rank,
            "index": int(idx),
            "bitstring": format(int(idx), f"0{n}b") if n > 0 else "",
            "probability": float(probs[idx]),
            "cut_value": float(cut_vals[idx]),
        })
    with open(outdir / "top_postselected_bitstrings.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(outdir / "top_postselected_bitstrings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def save_global_plots(
    steps: List[StepResult],
    outdir: Path,
    final_coeffs: Dict[int, float],
    total_possible_params: int,
    show_plots: bool = False,
) -> None:
    plot_dir = outdir / "plots"
    ensure_dir(plot_dir)

    visits = [s.visit_idx for s in steps]
    visited_js = [s.j for s in steps]
    losses = [s.loss for s in steps]
    expecteds = [s.expected_cut for s in steps]
    bests = [s.best_measured_cut for s in steps]
    keeps = [1 if s.kept else 0 for s in steps]
    improvements = [s.improvement_vs_previous for s in steps]

    coeff_vector = np.zeros(total_possible_params + 1, dtype=float)
    for j, val in final_coeffs.items():
        if 0 <= j <= total_possible_params:
            coeff_vector[j] = float(val)

    figs = [
        (range(1, total_possible_params + 1), coeff_vector[1:], "j", "coeficiente final", "Coeficientes Walsh finais", "final_coefficients_full_vector.png", "plot"),
        (visits, losses, "ordem de visita", "loss = - <H_cut>", "Loss por termo visitado", "loss_progress_by_visit.png", "plot"),
        (visits, expecteds, "ordem de visita", "cut", "Cut esperado", "expected_cut_progress_by_visit.png", "plot"),
        (visits, bests, "ordem de visita", "cut", "Cut da bitstring mais provável", "best_measured_cut_progress_by_visit.png", "plot"),
        (visits, keeps, "ordem de visita", "status", "Termos mantidos ou descartados", "kept_vs_discarded_by_visit.png", "step"),
        (visits, improvements, "ordem de visita", "Δ expected_cut", "Ganho marginal por termo adicionado", "marginal_gain_by_visit.png", "bar"),
        (visits, visited_js, "ordem de visita", "índice j visitado", "Mapa de visita dos índices", "visited_indices_map.png", "plot"),
    ]

    for x, y, xlabel, ylabel, title, filename, kind in figs:
        plt.figure(figsize=(8, 4))
        if kind == "bar":
            plt.bar(x, y)
        elif kind == "step":
            plt.step(x, y, where="mid")
            plt.yticks([0, 1], ["discard", "keep"])
        else:
            plt.plot(x, y, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=180)
        if show_plots:
            plt.show()
        plt.close()


def save_steps_table(steps: List[StepResult], outdir: Path) -> None:
    rows = [asdict(s) for s in steps]
    with open(outdir / "steps.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(outdir / "steps.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_sequential_walsh_maxcut(cfg: Dict[str, object]) -> Dict[str, object]:
    t0 = time.time()

    outdir = Path(cfg["OUTDIR"])
    ensure_dir(outdir)

    G = build_graph_from_cfg(cfg)
    n = G.number_of_nodes()
    reverse_bits = bool(cfg["REVERSE_BITS"])

    cut_vals = get_all_cut_values(G, reverse_bits=reverse_bits)
    graph_edges = graph_to_serializable_edges(G)
    graph_signature = graph_signature_from_edges(graph_edges)

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
    total_possible_params = (2**n) - 1

    no_improve_counter = 0
    prev_expected_cut = 0.0

    for visit_idx, j in enumerate(candidate_js, start=1):
        step, coeffs = train_one_unitary(
            j=j,
            visit_idx=visit_idx,
            frozen_coeffs=coeffs,
            G=G,
            n=n,
            reverse_bits=reverse_bits,
            cfg=cfg,
            outdir=outdir,
            cut_vals=cut_vals,
            previous_expected_cut=prev_expected_cut,
        )
        steps.append(step)

        improvement = step.improvement_vs_previous
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

    np.save(outdir / "final_coefficient_vector.npy", np.asarray(final_avec, dtype=float))
    np.save(outdir / "cut_values_vector.npy", np.asarray(cut_vals, dtype=float))
    with open(outdir / "trained_coefficients.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in sorted(coeffs.items())}, f, indent=2)

    save_global_plots(
        steps,
        outdir,
        final_coeffs=coeffs,
        total_possible_params=total_possible_params,
        show_plots=bool(cfg["SHOW_PLOTS"]),
    )
    save_steps_table(steps, outdir)
    save_final_distributions_and_states(outdir, final_metrics, cut_vals)

    best_expected_over_time = max((s.expected_cut for s in steps), default=0.0)
    best_measured_over_time = max((s.best_measured_cut for s in steps), default=0.0)
    final_coeff_vector = final_avec.tolist()
    active_indices = [int(k) for k, v in sorted(coeffs.items()) if k != 0 and abs(v) > 0.0]

    summary = {
        "config": cfg,
        "n_qubits_register": n,
        "graph_edges": graph_edges,
        "graph_signature": graph_signature,
        "graph_num_edges": int(G.number_of_edges()),
        "graph_density": float(nx.density(G)),
        "best_cut_bruteforce": float(best_cut),
        "best_bitstring_bruteforce": best_bitstring,
        "best_index_bruteforce": int(best_idx),
        "candidate_order": candidate_js,
        "total_possible_parameters": int(total_possible_params),
        "visited_parameters": int(len(steps)),
        "trained_coefficients": {str(k): float(v) for k, v in sorted(coeffs.items())},
        "final_coefficient_vector": final_coeff_vector,
        "active_indices": active_indices,
        "active_parameter_count": int(len(active_indices)),
        "compression_ratio_active_over_total": float(len(active_indices) / total_possible_params) if total_possible_params > 0 else 0.0,
        "final_loss": float(-final_metrics["expected_cut"]),
        "final_expected_cut": float(final_metrics["expected_cut"]),
        "best_expected_cut_over_evolution": float(best_expected_over_time),
        "final_best_measured_index": int(final_best_idx),
        "final_best_measured_bitstring": bits_to_str(final_best_bits),
        "final_best_measured_cut": float(final_best_cut),
        "best_measured_cut_over_evolution": float(best_measured_over_time),
        "final_success_probability_ancilla_1": float(final_metrics["success_probability_ancilla_1"]),
        "final_max_postselected_probability": float(final_metrics["max_postselected_probability"]),
        "gap_expected_to_bruteforce": float(best_cut - float(final_metrics["expected_cut"])),
        "gap_measured_to_bruteforce": float(best_cut - final_best_cut),
        "approx_ratio_measured": float(final_best_cut / best_cut) if abs(best_cut) > 1e-12 else 0.0,
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

    PRINT_OK_PLACEHOLDER
    print(json.dumps({
        "graph_signature": summary["graph_signature"],
        "best_cut_bruteforce": summary["best_cut_bruteforce"],
        "final_loss": summary["final_loss"],
        "final_expected_cut": summary["final_expected_cut"],
        "final_best_measured_bitstring": summary["final_best_measured_bitstring"],
        "final_best_measured_cut": summary["final_best_measured_cut"],
        "active_indices": summary["active_indices"],
    }, indent=2))