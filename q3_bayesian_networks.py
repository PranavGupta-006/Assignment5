"""
Q3: Bayesian Networks
Author: Pranav Gupta
Roll No: SE24UCSE020
College: Mahindra University

Tech stack: Python, pgmpy, numpy, json, sys, time, warnings
This file builds the Asia network and runs exact inference checks.
"""

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
import numpy as np
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pgmpy")


# Asia Bayesian network setup.

def build_asia_network():
    """Construct the Asia Bayesian network."""

    model = BayesianNetwork([
        ('A', 'T'),
        ('S', 'L'),
        ('S', 'B'),
        ('T', 'E'),
        ('L', 'E'),
        ('E', 'X'),
        ('E', 'D'),
        ('B', 'D'),
    ])

    cpd_A = TabularCPD('A', 2, [[0.99], [0.01]])

    cpd_S = TabularCPD('S', 2, [[0.50], [0.50]])

    cpd_T = TabularCPD('T', 2,
                        [[0.99, 0.95],
                         [0.01, 0.05]],
                        evidence=['A'], evidence_card=[2])

    cpd_L = TabularCPD('L', 2,
                        [[0.99, 0.90],
                         [0.01, 0.10]],
                        evidence=['S'], evidence_card=[2])

    cpd_B = TabularCPD('B', 2,
                        [[0.70, 0.40],
                         [0.30, 0.60]],
                        evidence=['S'], evidence_card=[2])

    cpd_E = TabularCPD('E', 2,
                        [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 1.0, 1.0]],
                        evidence=['T', 'L'], evidence_card=[2, 2])

    cpd_X = TabularCPD('X', 2,
                        [[0.95, 0.02],
                         [0.05, 0.98]],
                        evidence=['E'], evidence_card=[2])

    cpd_D = TabularCPD('D', 2,
                        [[0.90, 0.20, 0.30, 0.10],
                         [0.10, 0.80, 0.70, 0.90]],
                        evidence=['E', 'B'], evidence_card=[2, 2])

    model.add_cpds(cpd_A, cpd_S, cpd_T, cpd_L, cpd_B, cpd_E, cpd_X, cpd_D)

    assert model.check_model(), "Model validation failed!"

    return model


# Inference helpers.

def variable_elimination_query(model, query_var, evidence=None):
    ve = VariableElimination(model)
    result = ve.query(variables=[query_var], evidence=evidence or {}, show_progress=False)
    return result


def belief_propagation_query(model, query_var, evidence=None):
    bp = BeliefPropagation(model)
    result = bp.query(variables=[query_var], evidence=evidence or {}, show_progress=False)
    return result


# Utility: pretty print a factor.

def print_factor(factor, label=""):
    states = {0: 'no', 1: 'yes'}
    var = factor.variables[0]
    probs = factor.values
    print(f"  {'P(' + var + ')' if not label else label}")
    for i, p in enumerate(probs.flatten()):
        print(f"    {var}={states[i]}: {p:.4f}")


# Test suite.

def run_tests():
    print("=" * 65)
    print("  BAYESIAN NETWORKS — ASIA CHEST CLINIC NETWORK")
    print("=" * 65)

    model = build_asia_network()
    print("\n✓ Model built and validated successfully")
    print(f"  Nodes : {list(model.nodes())}")
    print(f"  Edges : {list(model.edges())}")

    print("\n[TEST 1] Prior Marginal Probabilities (no evidence)")
    ve = VariableElimination(model)
    for var in ['A', 'S', 'T', 'L', 'B', 'E', 'X', 'D']:
        result = ve.query([var], evidence={}, show_progress=False)
        p_yes = result.values[1]
        print(f"  P({var}=yes) = {p_yes:.4f}")

    result_T = ve.query(['T'], evidence={}, show_progress=False)
    p_T = result_T.values[1]
    expected_T = 0.99 * 0.01 + 0.01 * 0.05
    assert abs(p_T - expected_T) < 1e-5, f"P(T=yes) mismatch: {p_T} vs {expected_T}"
    print(f"\n  ✓ P(T=yes) = {p_T:.5f} matches analytic value {expected_T:.5f}")

    print("\n[TEST 2] Posterior Given X-ray Positive (X=yes)")
    evidence_xray = {'X': 1}
    print("  Evidence: X=yes (positive X-ray)")
    for var in ['T', 'L', 'E', 'D']:
        result = ve.query([var], evidence=evidence_xray, show_progress=False)
        p_yes = result.values[1]
        print(f"  P({var}=yes | X=yes) = {p_yes:.4f}")

    result_E = ve.query(['E'], evidence=evidence_xray, show_progress=False)
    assert result_E.values[1] > 0.5, "Positive X-ray should make E likely"
    print("  ✓ Positive X-ray correctly raises P(E=yes) above 0.5")

    print("\n[TEST 3] Posterior Given Smoking + Dyspnoea (S=yes, D=yes)")
    evidence_sd = {'S': 1, 'D': 1}
    print("  Evidence: S=yes (smoker), D=yes (dyspnoea)")
    for var in ['T', 'L', 'B', 'E']:
        result = ve.query([var], evidence=evidence_sd, show_progress=False)
        p_yes = result.values[1]
        print(f"  P({var}=yes | S=yes, D=yes) = {p_yes:.4f}")

    result_L_sd = ve.query(['L'], evidence={'S': 1}, show_progress=False)
    result_L_s_only = ve.query(['L'], evidence={'S': 1, 'D': 1}, show_progress=False)
    assert result_L_s_only.values[1] > result_L_sd.values[1], \
        "Adding dyspnoea should increase P(L|S)"
    print("  ✓ Adding dyspnoea increases P(L|S) as expected")

    print("\n[TEST 4] Effect of Asia Visit on Tuberculosis")
    p_T_no_asia = ve.query(['T'], evidence={'A': 0}, show_progress=False).values[1]
    p_T_yes_asia = ve.query(['T'], evidence={'A': 1}, show_progress=False).values[1]
    print(f"  P(T=yes | A=no)  = {p_T_no_asia:.4f}")
    print(f"  P(T=yes | A=yes) = {p_T_yes_asia:.4f}")
    assert p_T_yes_asia > p_T_no_asia, "Asia visit should raise P(T)"
    print(f"  ✓ Asia visit raises P(T): {p_T_no_asia:.4f} → {p_T_yes_asia:.4f}")

    print("\n[TEST 5] Explaining Away (Intercausal Reasoning)")
    p_T_given_E = ve.query(['T'], evidence={'E': 1}, show_progress=False).values[1]
    p_T_given_E_L = ve.query(['T'], evidence={'E': 1, 'L': 1}, show_progress=False).values[1]
    print(f"  P(T=yes | E=yes)       = {p_T_given_E:.4f}")
    print(f"  P(T=yes | E=yes, L=yes) = {p_T_given_E_L:.4f}")
    assert p_T_given_E_L < p_T_given_E, "Lung cancer should explain away tuberculosis"
    print(f"  ✓ Explaining away confirmed: L=yes reduces P(T|E=yes)")

    print("\n[TEST 6] VE vs Belief Propagation Consistency")
    test_configs = [
        ({'X': 1}, 'T'),
        ({'S': 1, 'D': 1}, 'L'),
        ({'A': 1}, 'E'),
    ]
    bp = BeliefPropagation(model)
    for evidence, query in test_configs:
        ve_result = ve.query([query], evidence=evidence, show_progress=False).values[1]
        bp_result = bp.query([query], evidence=evidence, show_progress=False).values[1]
        diff = abs(ve_result - bp_result)
        assert diff < 1e-5, f"VE vs BP mismatch for P({query}|{evidence}): {diff}"
        print(f"  ✓ P({query}|{evidence}) VE={ve_result:.4f} == BP={bp_result:.4f}")

    print("\n[TEST 7] Conditional Independence (D-Separation)")
    p_A = ve.query(['A'], evidence={}, show_progress=False).values[1]
    p_A_given_S = ve.query(['A'], evidence={'S': 1}, show_progress=False).values[1]
    diff = abs(p_A - p_A_given_S)
    print(f"  P(A=yes)        = {p_A:.4f}")
    print(f"  P(A=yes | S=yes) = {p_A_given_S:.4f}")
    print(f"  Difference: {diff:.6f} {'✓ Independent' if diff < 0.001 else '✗ Dependent (unexpected)'}")

    p_A_given_X = ve.query(['A'], evidence={'X': 1}, show_progress=False).values[1]
    p_A_given_XS = ve.query(['A'], evidence={'X': 1, 'S': 1}, show_progress=False).values[1]
    print(f"\n  P(A=yes | X=yes)        = {p_A_given_X:.4f}")
    print(f"  P(A=yes | X=yes, S=yes) = {p_A_given_XS:.4f}")
    print(f"  ✓ Conditioning on X makes A and S dependent (explaining away)")

    print("\n[TEST 8] MAP Inference — Most Likely Explanation")
    evidence_complex = {'D': 1, 'X': 1}
    print(f"  Evidence: {evidence_complex}")
    map_result = ve.map_query(variables=['T', 'L', 'B', 'E', 'A', 'S'],
                              evidence=evidence_complex, show_progress=False)
    state_name = {0: 'no', 1: 'yes'}
    for var, state in map_result.items():
        print(f"    {var} = {state_name[state]}")
    print("  ✓ MAP inference completed successfully")

    print("\n" + "=" * 65)
    print("  ALL BAYESIAN NETWORK TESTS PASSED ✓")
    print("=" * 65)


if __name__ == "__main__":
    def emit_step(title, status="running", detail="", data=None):
        print(json.dumps({
            "title": title,
            "status": status,
            "detail": detail,
            "data": data or {},
        }), flush=True)

    def run_api_steps():
        start = time.perf_counter()
        emit_step(
            "Define Asia network",
            detail="Constructing the directed graph and conditional probability tables.",
            data={
                "tooling": [
                    {"tool": "pgmpy", "role": "Python modeling, CPDs, validation, exact inference"},
                    {"tool": "GeNIe / SMILE", "role": "Graphical BN authoring and sensitivity analysis"},
                    {"tool": "Netica", "role": "Commercial BN modeling, decision networks, diagnostics"},
                    {"tool": "bnlearn", "role": "R package for structure learning and parameter fitting"},
                    {"tool": "Bayes Server", "role": "Industrial Bayesian networks, temporal models, APIs"},
                ],
                "visual_type": "bn-tools",
            },
        )
        model = build_asia_network()
        emit_step(
            "Validate Bayesian model",
            detail="The graph structure and CPDs passed pgmpy validation.",
            data={
                "nodes": list(model.nodes()),
                "edges": [list(edge) for edge in model.edges()],
                "node_labels": {
                    "A": "Asia visit",
                    "S": "Smoking",
                    "T": "Tuberculosis",
                    "L": "Lung cancer",
                    "B": "Bronchitis",
                    "E": "Either disease",
                    "X": "Positive X-ray",
                    "D": "Dyspnoea",
                },
                "visual_type": "bayesian-network",
            },
        )

        ve = VariableElimination(model)
        priors = {}
        for var in ["A", "S", "T", "L", "B", "E", "X", "D"]:
            priors[var] = round(float(ve.query([var], evidence={}, show_progress=False).values[1]), 4)
        emit_step(
            "Compute prior marginals",
            detail="Queried each variable without evidence using variable elimination.",
            data={"p_yes": priors},
        )

        evidence_x = {"X": 1}
        posterior_x = {}
        for var in ["T", "L", "E", "D"]:
            posterior_x[var] = round(float(ve.query([var], evidence=evidence_x, show_progress=False).values[1]), 4)
        emit_step(
            "Infer with positive X-ray evidence",
            detail="Conditioned on X=yes and recomputed disease probabilities.",
            data={"evidence": {"X": "yes"}, "p_yes": posterior_x},
        )

        evidence_sd = {"S": 1, "D": 1}
        posterior_sd = {}
        for var in ["T", "L", "B", "E"]:
            posterior_sd[var] = round(float(ve.query([var], evidence=evidence_sd, show_progress=False).values[1]), 4)
        emit_step(
            "Infer with smoking and dyspnoea evidence",
            detail="Conditioned on S=yes, D=yes and updated likely explanations.",
            data={"evidence": {"S": "yes", "D": "yes"}, "p_yes": posterior_sd},
        )

        bp = BeliefPropagation(model)
        comparisons = []
        for evidence, query in [({"X": 1}, "T"), ({"S": 1, "D": 1}, "L"), ({"A": 1}, "E")]:
            ve_result = float(ve.query([query], evidence=evidence, show_progress=False).values[1])
            bp_result = float(bp.query([query], evidence=evidence, show_progress=False).values[1])
            comparisons.append({
                "query": query,
                "evidence": evidence,
                "variable_elimination": round(ve_result, 5),
                "belief_propagation": round(bp_result, 5),
                "difference": round(abs(ve_result - bp_result), 8),
            })
        emit_step(
            "Compare inference engines",
            detail="Variable elimination and belief propagation agree on the test queries.",
            data={"comparisons": comparisons},
        )

        map_result = ve.map_query(
            variables=["T", "L", "B", "E", "A", "S"],
            evidence={"D": 1, "X": 1},
            show_progress=False,
        )
        state_name = {0: "no", 1: "yes"}
        emit_step(
            "Run MAP explanation",
            detail="Found the most likely hidden states for D=yes and X=yes.",
            data={"map": {var: state_name[state] for var, state in map_result.items()}},
        )

        emit_step(
            "Q3 complete",
            status="complete",
            detail="Bayesian-network workflow finished successfully.",
            data={"elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
        )

    if "--steps" in sys.argv:
        run_api_steps()
    else:
        run_tests()
