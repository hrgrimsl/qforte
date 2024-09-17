import qforte as qf
from qforte.system.molecular_info import System


def create_XY_chain(n: int):
    """Creates a Spin-1/2 XY Model hamiltonian with
    periodic boundary conditions

    n: int
        Number of lattice sites    
    """
    
    try:
        assert n >= 2
    except:
        return ValueError("XY Hamiltonian requires at least 2 qubits.")
        
    XY_chain = System()
    XY_chain.hamiltonian = qf.QubitOperator()

    circuit = [(-1, f"X_{i} X_{(i+1)%n}") for i in range(n)]
    circuit += [(-1, f"Y_{i} Y_{(i+1)%n}") for i in range(n)]
    
    for coeff, op_str in circuit:
        XY_chain.hamiltonian.add(coeff, qf.build_circuit(op_str))

    XY_chain.hf_reference = [0] * n

    return XY_chain

def create_TFIM(n: int, h: float, J: float):
    """Creates a 1D Transverse Field Ising Model hamiltonian with
    open boundary conditions, i.e., no interaction between the
    first and last spin sites.

    n: int
        Number of lattice sites

    h: float
        Strength of magnetic field

    j: float
        Interaction strength
    """

    TFIM = System()
    TFIM.hamiltonian = qf.QubitOperator()

    circuit = [(-h, f"Z_{i}") for i in range(n)]
    circuit += [(-J, f"X_{i} X_{i+1}") for i in range(n - 1)]

    for coeff, op_str in circuit:
        TFIM.hamiltonian.add(coeff, qf.build_circuit(op_str))

    TFIM.hf_reference = [0] * n

    return TFIM
