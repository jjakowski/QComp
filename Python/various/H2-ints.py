

import psi4
import numpy as np

#--- geometry
HH_geometry = psi4.geometry ("""
0 1
H   0.000000   0.0      0.0 
H   0.8        0.0      0.0 
symmetry c1
no_reorient
no_com
""")

#------------------------------------------------------------------------------------------
# Get MO coefficients from SCF wavefunction
#basis = '6-311G', '6-31G(d)', 'sto-3g',

psi4.set_options({'basis':        'sto-3g',
                  'scf_type':     'pk',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

#  Solve SCF: # Get the SCF wavefunction & energies  
scf_e  , scf_wfn   = psi4.energy('scf', return_wfn=True,molecule=HH_geometry) 

#nuclear repulsion energy
E_nucl   = HH_geometry.nuclear_repulsion_energy()
print("Nuclear repulsion :", E_nucl)

# read in matrices from SCF  (a.o.) :
S    =  np.array(scf_wfn.S())   # overlap
Fa   =  np.array(scf_wfn.Fa())  # fock
H    =  np.array(scf_wfn.H())   # Core Hamiltonian
Ca   =  np.array(scf_wfn.Ca())  # alpha MO
Cb   =  np.array(scf_wfn.Cb())  # beta  MO

#---integrals:             
mints   = psi4.core.MintsHelper(scf_wfn.basisset()) 
Ints    = np.array(mints.ao_eri())  
print('AO itngetrals: ', Ints)






