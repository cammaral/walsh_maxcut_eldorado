import pennylane as qml
import pennylane.numpy as np


#---- Expansões Binárias ----#
def binary_expansion(j, n):
    return [int(bit) for bit in bin(j)[2:].zfill(n)]

def dyadic_expansion(x, n):
    if 0 <= x < 1:
        expansion = []
        for _ in range(n):
            bit = int(x * 2)
            expansion.append(bit)
            x = x * 2 - bit
        return expansion


#---- Gera CNOT + RZ ----#
def Rj_circuit(avec, n, list_j=None):
    def analisar_bitstring(bitstring):
        posicoes_de_1 = []
        posicao_do_ultimo_1 = -1
        for i, bit in enumerate(bitstring):
            if bit == '1':
                posicoes_de_1.append(i + 1)   # registrador nos wires 1..n
                posicao_do_ultimo_1 = i + 1
        return posicoes_de_1, posicao_do_ultimo_1

    if list_j is None:
        list_j = range(1, len(avec))

    for i in list_j:
        if i == 0:
            continue

        theta = 2 * avec[i]
        jbin = bin(i)[2:].zfill(n)[::-1]
        p1, pu1 = analisar_bitstring(jbin)

        if len(p1) == 1:
            qml.RZ(theta, wires=pu1)
        else:
            for p in p1[:-1]:
                qml.CNOT(wires=[p, pu1])

            qml.RZ(theta, wires=pu1)

            for p in p1[:-1][::-1]:
                qml.CNOT(wires=[p, pu1])


def postselect(state, n):
    state = qml.math.reshape(state, (2, 2**n))
    reg_state = state[1]   # ancilla = 1
    norm = qml.math.sqrt(qml.math.sum(qml.math.conj(reg_state) * reg_state))
    return reg_state / norm if not qml.math.allclose(norm, 0) else reg_state


def prob(state):
    norm = qml.math.sqrt(qml.math.sum(qml.math.conj(state) * state))
    return qml.math.abs(state / norm) ** 2

#--- Rodar o Circuito ---#

def run_circuit(avec, n, er=1e-6):
    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev)
    def circuit(avec, er=1e-6):
        avec = er * np.array(avec)

        # ancilla + registrador em superposição
        for i in range(n + 1):
            qml.Hadamard(wires=i)

        # controlled-U_rest
        qml.ctrl(Rj_circuit, control=0)(avec, n)

        # termo a0 como fase relativa na ancilla
        qml.PhaseShift(-avec[0], wires=0)

        # bloco final da ancilla
        qml.Hadamard(wires=0)
        qml.PhaseShift(-np.pi / 2, wires=0)

        return qml.state()
    return postselect(circuit(avec, er=er), n)

def bits_to_state(n, bits):
    if len(bits) != n:
        raise ValueError("O tamanho de bits deve ser igual a n.")
    
    if any(b not in [0, 1] for b in bits):
        raise ValueError("bits deve conter apenas 0 e 1.")
    
    bitstring = "".join(map(str, bits))
    labels = [f"{i:0{n}b}" for i in range(2**n)]
    
    vec = np.zeros(2**n, dtype=int)
    vec[int(bitstring, 2)] = 1
    
    return bitstring, labels, vec