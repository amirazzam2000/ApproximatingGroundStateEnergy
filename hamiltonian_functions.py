from pennylane import qchem
import pennylane as qml
import numpy as np
import time 
import matplotlib.pyplot as plt



def double_excitation_circute(params, wires, excitations, hf_state):
    '''
    Compute gradients for all double excitations.
    '''
    qml.BasisState(hf_state, wires=wires)

    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        else:
            qml.SingleExcitation(params[i], wires=excitation)

def single_excitation_circute(params, wires, excitations, gates_select, params_select, hf_state):
    '''
    Computes gradients for all single excitations.
    '''
    qml.BasisState(hf_state, wires=wires)

    for i, gate in enumerate(gates_select):
        if len(gate) == 4:
            qml.DoubleExcitation(params_select[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params_select[i], wires=gate)

    for i, gate in enumerate(excitations):
        if len(gate) == 4:
            qml.DoubleExcitation(params[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params[i], wires=gate)


class HamiltonianFinder:
    '''
    This class is used to define the Hamiltonain of a molecule and find its ground state energy. 
    '''

    energies = {}

    def groundStateFinder(
        self,
        active_electrons ,
        active_orbitals ,
        name,
        symbols , 
        coordinates, 
        basis_set = 'sto-3g',
        mult = 1,
        charge = 0,
        mapping = 'bravyi_kitaev',
        verbose = 2
        ):
        '''
        This method finds the ground state energy for the molecule constructed by the parameters of this method.

        : param active_electrons: the number of active electrons in the molecule 
        : param active_orbitals: the number of active orbitals in the molecule 
        : param name: the name of the molecule 
        : param symbols: the symbols of the atoms constructing the molecules 
        : param coordinates: the coordinates of the atoms constructing the molecules. These should be specified in the same order as their corresponding symbols.
        : param basis_set: the basis set used to define the orbital distribution.  
        : param mult: the multiplicity of the molecule
        : param charge: the net charge of the molecule 
        : param mapping: the fermionic-qubit mapping used to map this molecule.
        : param verbose: the verbos of the output. 

        :returns: the Hamiltonian circuit as a string, and the energy of that Hamiltonian
        '''

        if(verbose>= 3):
            print("parameters: ")
            print("active electrons: ", active_electrons)
            print("active orbitals: ", active_orbitals)
            print("name: ", name)
            print("symbols: ", symbols)
            print("coordinates: ", coordinates)
            print("multiplicity: ", mult)
            print("charge: ", charge)
            print("mapping: ", mapping)
        


        H, qubits = qchem.molecular_hamiltonian(
            symbols, 
            coordinates,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            name=name,
            charge=charge,
            mult=mult,
            basis= basis_set,
            package='pyscf',
            mapping=mapping
            )

        # caclulate the single and double excitations

        singles, doubles = qchem.excitations(active_electrons, qubits)
        if verbose >= 2:
            print(f"Total number of excitations = {len(singles) + len(doubles)}")

        hf_state = qchem.hf_state(active_electrons, qubits)

        dev = qml.device("default.qubit", wires=qubits)
        cost_fn = qml.ExpvalCost(double_excitation_circute, H, dev, optimize=True) #, kwargs={"hf_state": hf_state})

        circuit_gradient = qml.grad(cost_fn, argnum=0)

        params = [0.0] * len(doubles)
        grads = circuit_gradient(
            params, 
            excitations=doubles, 
            hf_state=hf_state
            )

        if verbose >= 3:
            for i in range(len(doubles)):
                print(f"Excitation : {doubles[i]}, Gradient: {grads[i]}")


        doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
        if verbose >= 3:
            print(doubles_select)

        opt = qml.GradientDescentOptimizer(stepsize=0.5)

        params_doubles = np.zeros(len(doubles_select))

        for n in range(20):
            params_doubles = opt.step(cost_fn, params_doubles, excitations=doubles_select, hf_state=hf_state) 


        cost_fn = qml.ExpvalCost(single_excitation_circute, H, dev, optimize=True)
        circuit_gradient = qml.grad(cost_fn, argnum=0)
        params = [0.0] * len(singles)

        grads = circuit_gradient(
            params,
            excitations=singles,
            gates_select=doubles_select,
            params_select=params_doubles,
            hf_state = hf_state
        )

        if verbose >= 3:
            for i in range(len(singles)):
                print(f"Excitation : {singles[i]}, Gradient: {grads[i]}")
            
        singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]
        
        if verbose >= 3:
            print(singles_select)


        H_sparse = qml.utils.sparse_hamiltonian(H)

        opt = qml.GradientDescentOptimizer(stepsize=0.5)

        excitations = doubles_select + singles_select

        params = np.zeros(len(excitations))

        @qml.qnode(dev, diff_method="parameter-shift")
        def hamiltonian_circuit(params, hf_state):
            qml.BasisState(hf_state, wires=range(qubits))

            for i, excitation in enumerate(excitations):
                if len(excitation) == 4:
                    qml.DoubleExcitation(params[i], wires=excitation)
                elif len(excitation) == 2:
                    qml.SingleExcitation(params[i], wires=excitation)

            return qml.expval(qml.SparseHamiltonian(H_sparse, wires=range(qubits)))


        self.energies = {'labels' : [], 'values' : []}
        for n in range(20):
            t1 = time.time()
            params, energy = opt.step_and_cost(hamiltonian_circuit, params, hf_state=hf_state)
            t2 = time.time()
            self.energies['values'].append(energy.numpy())
            self.energies['labels'].append(n)
            if verbose >= 3:
                print("n = {:},  E = {:.8f} H, t = {:.2f} s".format(n, energy, t2 - t1))


        print()
        if verbose >= 2:
            print("minimum energy: ", energy)
        print()

        return energy.numpy(), H

    def printEnergyEvelotion(self):
        '''
        creates a plot showing the evolution of the energy in optimization process. 
        '''
        print('labels: ', self.energies['labels'] )
        print('values: ', self.energies['values'] )
        plt.plot(self.energies['labels'] ,self.energies['values'])

        # naming the x axis
        plt.xlabel('Steps')
        # naming the y axis
        plt.ylabel('Energy')
        
        # giving a title to my graph
        plt.title('Energy Evelotion')

        plt.show()