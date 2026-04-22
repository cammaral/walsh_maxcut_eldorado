from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from train_maxcut_walsh_article import CONFIG, run_sequential_walsh_maxcut


BENCHMARK_CONFIG = {
    "vertex_sizes": [4, 6, 8],
    "instances_per_size": 10,
    "train_seeds": list(range(10)),
    "families": [
        {
            "name": "unweighted",
            "com_peso": False,
            "weight_mode": "unit",
            "probabilities": {
                4: [1.00, 0.85, 0.70, 0.55, 0.40],
                6: [1.00, 0.80, 0.65, 0.50, 0.35],
                8: [1.00, 0.75, 0.60, 0.45, 0.30],
            },
        },
        {
            "name": "weighted_positive",
            "com_peso": True,
            "weight_mode": "positive",
            "peso_min": 1,
            "peso_max": 3,
            "probabilities": {
                4: [1.00, 0.85, 0.70, 0.55, 0.40],
                6: [1.00, 0.80, 0.65, 0.50, 0.35],
                8: [1.00, 0.75, 0.60, 0.45, 0.30],
            },
        },
        {
            "name": "weighted_signed",
            "com_peso": True,
            "weight_mode": "signed",
            "peso_min": -3,
            "peso_max": 3,
            "probabilities": {
                4: [1.00, 0.85, 0.70, 0.55, 0.40],
                6: [1.00, 0.80, 0.65, 0.50, 0.35],
                8: [1.00, 0.75, 0.60, 0.45, 0.30],
            },
        },
    ],
    "base_train_cfg": {
        "ER": 1.0,
        "MAX_TERMS": None,
        "A_MAX": math.pi / 4,
        "INIT_BETA": 0.35,
        "MAX_EPOCHS": 100,
        "LR": 0.20,
        "FD_EPS": 1e-2,
        "LOSS_TOL_LAST5": 1e-3,
        "PATIENCE_LAST5": 5,
        "COEF_ZERO_TOL": 1e-3,
        "IMPROVEMENT_TOL": 1e-3,
        "SATURATION_PATIENCE": 100,
        "INDEX_STRATEGY": "ascending",
        "ENABLE_BACKWARD_PRUNING": False,
        "BACKWARD_PASSES": 1,
        "PRUNE_IMPROVEMENT_TOL": 1e-6,
        "SHOW_PLOTS": False,
        "INIT_BETA_JITTER": 0.05,
        "REVERSE_BITS": False,
    },
    "outdir": "benchmark_results_walsh_article",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def graph_signature_from_nx(G: nx.Graph) -> str:
    edges = []
    for u, v, data in sorted(G.edges(data=True), key=lambda t: (min(t[0], t[1]), max(t[0], t[1]))):
        edges.append((int(min(u, v)), int(max(u, v)), round(float(data.get("weight", 1.0)), 12)))
    payload = json.dumps(edges, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_edges_from_nx(G: nx.Graph) -> List[List[float]]:
    edges = []
    for u, v, data in sorted(G.edges(data=True), key=lambda t: (min(t[0], t[1]), max(t[0], t[1]))):
        edges.append([int(min(u, v)), int(max(u, v)), float(data.get("weight", 1.0))])
    return edges


def sample_graph(n_vertices: int, prob: float, family: Dict[str, object], graph_seed: int) -> nx.Graph:
    rng = random.Random(graph_seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    if abs(prob - 1.0) < 1e-12:
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                G.add_edge(u, v)
    else:
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if rng.random() < prob:
                    G.add_edge(u, v)

    if family["weight_mode"] == "unit":
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
    elif family["weight_mode"] == "positive":
        wmin = int(family.get("peso_min", 1))
        wmax = int(family.get("peso_max", 3))
        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.randint(wmin, wmax))
    elif family["weight_mode"] == "signed":
        wmin = int(family.get("peso_min", -3))
        wmax = int(family.get("peso_max", 3))
        for u, v in G.edges():
            w = 0
            while w == 0:
                w = rng.randint(wmin, wmax)
            G[u][v]["weight"] = float(w)
    else:
        raise ValueError(f"Modo de peso desconhecido: {family['weight_mode']}")

    return G


def generate_unique_graph_bank(
    n_vertices: int,
    family: Dict[str, object],
    instances_per_size: int,
) -> List[Dict[str, object]]:
    probs = family["probabilities"][n_vertices]
    used_signatures = set()
    out = []

    # garante 1 completo
    full_seed = 10_000 + n_vertices * 101 + len(family["name"])
    G_full = sample_graph(n_vertices, 1.0, family, full_seed)
    sig_full = graph_signature_from_nx(G_full)
    used_signatures.add(sig_full)
    out.append({
        "graph_id": 0,
        "prob_aresta": 1.0,
        "graph_seed": full_seed,
        "graph_signature": sig_full,
        "graph_edges": graph_edges_from_nx(G_full),
    })

    graph_id = 1
    attempts = 0
    while len(out) < instances_per_size:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError(f"Não consegui gerar grafos únicos suficientes para n={n_vertices}, família={family['name']}")

        prob = probs[(graph_id - 1) % len(probs)]
        if abs(prob - 1.0) < 1e-12:
            prob = probs[(graph_id) % len(probs)]
        graph_seed = 100_000 + 1000 * n_vertices + 100 * graph_id + attempts + len(family["name"]) * 17
        G = sample_graph(n_vertices, prob, family, graph_seed)
        sig = graph_signature_from_nx(G)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        out.append({
            "graph_id": graph_id,
            "prob_aresta": float(prob),
            "graph_seed": int(graph_seed),
            "graph_signature": sig,
            "graph_edges": graph_edges_from_nx(G),
        })
        graph_id += 1

    return out


def aggregate_group(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    return {
        "family": rows[0]["family"],
        "n_vertices": rows[0]["n_vertices"],
        "n_runs": len(rows),
        "mean_final_expected_cut": mean(float(r["final_expected_cut"]) for r in rows),
        "std_final_expected_cut": pstdev(float(r["final_expected_cut"]) for r in rows),
        "mean_final_best_measured_cut": mean(float(r["final_best_measured_cut"]) for r in rows),
        "std_final_best_measured_cut": pstdev(float(r["final_best_measured_cut"]) for r in rows),
        "mean_gap_measured_to_bruteforce": mean(float(r["gap_measured_to_bruteforce"]) for r in rows),
        "std_gap_measured_to_bruteforce": pstdev(float(r["gap_measured_to_bruteforce"]) for r in rows),
        "mean_approx_ratio_measured": mean(float(r["approx_ratio_measured"]) for r in rows),
        "std_approx_ratio_measured": pstdev(float(r["approx_ratio_measured"]) for r in rows),
        "mean_active_parameter_count": mean(float(r["active_parameter_count"]) for r in rows),
        "std_active_parameter_count": pstdev(float(r["active_parameter_count"]) for r in rows),
        "mean_compression_ratio": mean(float(r["compression_ratio_active_over_total"]) for r in rows),
        "std_compression_ratio": pstdev(float(r["compression_ratio_active_over_total"]) for r in rows),
        "mean_success_probability": mean(float(r["final_success_probability_ancilla_1"]) for r in rows),
        "std_success_probability": pstdev(float(r["final_success_probability_ancilla_1"]) for r in rows),
    }


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_benchmark() -> Dict[str, object]:
    outdir = Path(BENCHMARK_CONFIG["outdir"])
    ensure_dir(outdir)

    all_run_rows: List[Dict[str, object]] = []
    aggregate_rows: List[Dict[str, object]] = []
    graph_bank_rows: List[Dict[str, object]] = []

    for family in BENCHMARK_CONFIG["families"]:
        family_name = family["name"]
        family_dir = outdir / family_name
        ensure_dir(family_dir)

        for n_vertices in BENCHMARK_CONFIG["vertex_sizes"]:
            graph_bank = generate_unique_graph_bank(
                n_vertices=n_vertices,
                family=family,
                instances_per_size=BENCHMARK_CONFIG["instances_per_size"],
            )

            # salva banco de grafos para auditoria
            with open(family_dir / f"graph_bank_n{n_vertices}.json", "w", encoding="utf-8") as f:
                json.dump(graph_bank, f, indent=2)

            for graph_info in graph_bank:
                graph_bank_rows.append({
                    "family": family_name,
                    "n_vertices": n_vertices,
                    **{k: v for k, v in graph_info.items() if k != "graph_edges"},
                })

                for train_seed in BENCHMARK_CONFIG["train_seeds"]:
                    cfg = copy.deepcopy(CONFIG)
                    cfg.update(copy.deepcopy(BENCHMARK_CONFIG["base_train_cfg"]))
                    cfg["N_VERTICES"] = n_vertices
                    cfg["PROB_ARESTA"] = graph_info["prob_aresta"]
                    cfg["GRAPH_SEED"] = graph_info["graph_seed"]
                    cfg["GRAPH_EDGES"] = graph_info["graph_edges"]
                    cfg["COM_PESO"] = bool(family["com_peso"])
                    if family["weight_mode"] == "positive":
                        cfg["PESO_MIN"] = int(family["peso_min"])
                        cfg["PESO_MAX"] = int(family["peso_max"])
                    elif family["weight_mode"] == "signed":
                        cfg["PESO_MIN"] = int(family["peso_min"])
                        cfg["PESO_MAX"] = int(family["peso_max"])
                    else:
                        cfg["PESO_MIN"] = 1
                        cfg["PESO_MAX"] = 1
                    cfg["TRAIN_SEED"] = int(train_seed)

                    run_name = f"n{n_vertices}_graph{graph_info['graph_id']:02d}_trainseed{train_seed:02d}"
                    cfg["OUTDIR"] = str(family_dir / f"runs_{n_vertices}" / run_name)

                    summary = run_sequential_walsh_maxcut(cfg)

                    row = {
                        "family": family_name,
                        "n_vertices": n_vertices,
                        "graph_id": graph_info["graph_id"],
                        "graph_seed": graph_info["graph_seed"],
                        "graph_signature": graph_info["graph_signature"],
                        "prob_aresta": graph_info["prob_aresta"],
                        "train_seed": train_seed,
                        "best_cut_bruteforce": summary["best_cut_bruteforce"],
                        "final_expected_cut": summary["final_expected_cut"],
                        "final_best_measured_cut": summary["final_best_measured_cut"],
                        "gap_expected_to_bruteforce": summary["gap_expected_to_bruteforce"],
                        "gap_measured_to_bruteforce": summary["gap_measured_to_bruteforce"],
                        "approx_ratio_measured": summary["approx_ratio_measured"],
                        "total_possible_parameters": summary["total_possible_parameters"],
                        "visited_parameters": summary["visited_parameters"],
                        "active_parameter_count": summary["active_parameter_count"],
                        "compression_ratio_active_over_total": summary["compression_ratio_active_over_total"],
                        "final_success_probability_ancilla_1": summary["final_success_probability_ancilla_1"],
                        "elapsed_seconds": summary["elapsed_seconds"],
                        "outdir": cfg["OUTDIR"],
                    }
                    all_run_rows.append(row)

            family_rows = [r for r in all_run_rows if r["family"] == family_name and r["n_vertices"] == n_vertices]
            aggregate_rows.append(aggregate_group(family_rows))

    save_csv(all_run_rows, outdir / "all_runs.csv")
    save_csv(aggregate_rows, outdir / "aggregated_by_family_and_size.csv")
    save_csv(graph_bank_rows, outdir / "graph_bank_catalog.csv")

    result = {
        "benchmark_config": BENCHMARK_CONFIG,
        "n_runs": len(all_run_rows),
        "all_runs": all_run_rows,
        "aggregates": aggregate_rows,
        "graph_bank_catalog": graph_bank_rows,
    }
    with open(outdir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    out = run_benchmark()
    print(json.dumps({
        "n_runs": out["n_runs"],
        "aggregates": out["aggregates"],
    }, indent=2))
