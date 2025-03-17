import qiskit.tools.jupyter
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from vqa_poisson import VQAforPoisson

# Function to handle deprecated imports
def safe_import():
    try:
        import rustworkx as rx
    except ImportError:
        import retworkx as rx

safe_import()

def experiment(bc, num_trials, num_qubits_list, num_layers, qins):
    print('-----------' + bc + ' boundary condition --------------')
    data = {'num_qubits': [], 'obj_count': [], 'circ_count': [], 'iter_count': [], 'err': [], 'params': [], 'q_sol': [], 'cl_sol': []}
    for num_qubits in tqdm(num_qubits_list):
        print('-------------------------')
        print('num_qubits:', num_qubits)
        oracle_f = QuantumCircuit(num_qubits)
        oracle_f.x(num_qubits-1)
        oracle_f.h(oracle_f.qubits)
        vqa = VQAforPoisson(num_qubits, num_layers, bc, oracle_f=oracle_f, qinstance=qins)
        obj_counts, circ_counts, iter_counts, err, params, q_sol = [], [], [], [], [], []
        for seed in range(num_trials):
            np.random.seed(seed)
            x0 = list(4*np.pi*np.random.rand(vqa.num_params))
            res = vqa.minimize(x0, method='bfgs', save_logs=True)
            obj_counts.append(vqa.objective_counts)
            circ_counts.append(vqa.circuit_counts)
            iter_counts.append(len(vqa.objective_count_logs))
            trace_val = np.real(vqa.get_errors(res['x'])['trace'])
            if trace_val >= 0:
                err.append(np.sqrt(1 - trace_val))
            else:
                err.append(np.nan)
            params.append(res['x'])
            q_sol.append(vqa.get_sol(res['x']).real)
            print('trial:', seed, 'Err.:', err[-1])
        data['num_qubits'].append(num_qubits)
        data['obj_count'].append(obj_counts)
        data['circ_count'].append(circ_counts)
        data['iter_count'].append(iter_counts)
        data['err'].append(err)
        data['params'].append(params)
        data['q_sol'].append(q_sol)
        data['cl_sol'].append(vqa.get_cl_sol().real)
    return data

t0 = time.time()
optimizer = 'bfgs'
num_layers = 5
num_trials = 1
num_qubits_list = [4]
qins = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=42)
data_p = experiment('Periodic', num_trials, num_qubits_list, num_layers, qins)

def plot_solution_vectors(q_sol, cl_sol):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(q_sol, label='quantum', color='black')
    ax.plot(cl_sol, label='classical', color='black', linestyle='dashed')
    ax.legend()
    ax.set_xlabel('Node number')
    ax.set_ylabel('Components of solution')
    cnorm = np.linalg.norm(q_sol)
    qnorm = np.linalg.norm(cl_sol)
    ax.text(0.55, 0.65, 'Norm (quantum) = %.1f'%(qnorm), transform=ax.transAxes)
    ax.text(0.55, 0.55, 'Norm (classical) = %.1f'%(cnorm), transform=ax.transAxes)
    return fig, ax

idx1, idx2 = 0, 0
print('Periodic boundary condition, num_qubits:', data_p['num_qubits'][idx1])
q_sol = data_p['q_sol'][idx1][idx2]
cl_sol = data_p['cl_sol'][idx1]
plot_solution_vectors(q_sol, cl_sol)
print('elapsed time: %.2e'%(time.time() - t0))