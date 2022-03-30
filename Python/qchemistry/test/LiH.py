
#from  qcpy import  *
from qchemistry.qcpy import  *

import psi4
import numpy as np


# Initialize structure
LiH_geometry = psi4.geometry ("""
0 1
Li   0.000000   0.0      0.0
H   0.0        0.0    1.598436 
symmetry c1
""")

#basis = '6-311G'
psi4.set_options({'basis':        'sto-3g-jacek',
                  'scf_type':     'pk',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})


#-----------------------------------------------
# Solve scf:
#-----------------------------------------------

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True,molecule=LiH_geometry)

print("E(SCF):", scf_e, scf_wfn)

print("")
print('Alpha orbital energies: ', np.array(scf_wfn.epsilon_a()))
print('Beta  orbital energies: ', np.array(scf_wfn.epsilon_b()))
#print(scf_wfn.doccpi() )

print("")
print("N-alpha-electrons:  %d "  % scf_wfn.nalpha())
print("N-beta--electrons:  %d "  % scf_wfn.nbeta())

print("")
print("Alpha occupations: ", np.asarray(scf_wfn.occupation_a()))
print("Beta  occupations: ", np.asarray(scf_wfn.occupation_b()))

#nuclear repulsion energy
E_nucl = LiH_geometry.nuclear_repulsion_energy()
print("")
print("Nuclear repulsion:", E_nucl)



#
#-----------------------------------------------
#  Electron Repulsion integrals:¶
#-----------------------------------------------

# Get MO coefficients from SCF wavefunction

# ==> ERIs <==
# Create instance of MintsHelper class:
mints = psi4.core.MintsHelper(scf_wfn.basisset())


nbf = mints.nbf()           # number of basis functions
nso = 2 * nbf               # number of spin orbitals
nalpha = scf_wfn.nalpha()   # number of alpha electrons
nbeta = scf_wfn.nbeta()     # number of beta electrons
nocc = nalpha + nbeta       # number of occupied orbitals
nvirt = 2 * nbf - nocc      # number of virtual orbitals
list_occ_alpha= np.asarray(scf_wfn.occupation_a())
list_occ_beta = np.asarray(scf_wfn.occupation_b())


# Get orbital energies 
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)


# Get orbital coefficients: 
Ca = np.asarray(scf_wfn.Ca())              
Cb = np.asarray(scf_wfn.Cb())              
C = np.block([ 
                [      Ca           ,   np.zeros_like(Cb) ],
                [np.zeros_like(Ca)  ,          Cb         ]
             ])


#-----------------------------------------------
#  Build Hamiltonian in second quantization:¶
#-----------------------------------------------

#-------- 0-body term: 
Hamiltonian_0body = E_nucl


#-------- 1-body term:
# Read from file:
#   H        =  np.array(scf_wfn.H())      
# or build core Hamiltonian
#
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H_core_ao = T + V
Hamiltonian_1body  =  get_Hamiltonian_1body(H_core_ao,C)  


#--------- 2-body term
#   H_2body =  0.25* \sum
#Hamiltonian_2body = 0.25*gmo

# Get the two electron integrals using MintsHelper
Ints = np.array(mints.ao_eri())
gao  = get_gao(Ints)   #  chemists ->  physicists notation
gmo  = get_Hamiltonian_2body(gao,C)  ;  # C is over spin-orbitals 
Hamiltonian_2body = gmo 
#Hamiltonian_2body = get_Hamiltonian_2body(gao,get_spin_blocked(Ca))



#-----------------------------------------------
#  Now go ahead with FCI calcs  with two different Hamilaotnians:
#  1)  Full 2nd quantization hamiltonian
#  2)  Effective (frozen core) 2nd quantization  hamiltonian 
#
#-----------------------------------------------


#
# -- list of MOs
#--LiH
#

MSO_alpha_occ  = np.array([0,1])
MSO_alpha_virt = np.array([2,3,4,5])

MSO_beta_occ   = np.array([6,7]) 
MSO_beta_virt  = np.array([8,9,10,11])


MSO_occ =  np.append(MSO_alpha_occ,  MSO_beta_occ)
MSO_virt=  np.append(MSO_alpha_virt, MSO_beta_virt)

n_occ  = len(MSO_occ )
n_virt = len(MSO_virt)




#----------------------------------------------------------------
#  FCI energy
#
#----------------------------------------------------------------


from scipy.linalg  import eigh
from scipy.sparse import coo_matrix
import scipy.sparse.linalg 



help(get_FCI_hamiltonian)     

table_no_of_qbits         =  np.array([]) ;   # number of qubits / active MSO.
table_full_2nd_quant      =  np.array([]) ;   # save data from full 2nd quantization FCI
table_reduced_2nd_quant   =  np.array([]) ;   # save data from reduced (frozen core)  Hamitloanian
  

print("==================================================================================");
#---LiH (12-quibits)
print("     12-qubits: Full Hamiltonian,  no-frozen core, all  vritual MSO included");
MSO_occ_tmp         = [0,1,        6,7  ] 
MSO_frozen_list_tmp = [                 ]
MSO_active_list_tmp = [0, 1,2,3,4,5, 6, 7,8,9,10,11 ]

[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                          MSO_occ_tmp,
                          MSO_frozen_list_tmp,
                          MSO_active_list_tmp,
                          Hamiltonian_0body,Hamiltonian_1body,gmo) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)



E_FCI =  FCI_eigenvalues[0]
E_FCI_corr = E_FCI - scf_e
print('Occupied MSO: ',MSO_occ_tmp)
print('Frozen   MSO: ',MSO_frozen_list_tmp)
print('Active   MSO: ',MSO_active_list_tmp)
print('E_HF           = ',scf_e)
print('E_FCI          = ',E_FCI)
print('E_FCI_corr     = ',E_FCI_corr)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print("  ");

#MSO_virt_tmp =  [ item for  item in  MSO_active_list_tmp  if item not in MSO_occ_tmp ]
#E_MP2_corr = get_MP2_corr(MSO_occ_tmp,MSO_virt_tmp,eps,gmo,False)
E_MP2_corr = get_MP2_corr(MSO_occ    ,MSO_virt    ,eps,gmo,False)
print('MP2 correlation energy: ', E_MP2_corr)
print('MP2 total energy:       ', scf_e +E_MP2_corr)


table_no_of_qbits         =   np.append(table_no_of_qbits, len(MSO_active_list_tmp)) ; 
table_full_2nd_quant      =   np.append(table_full_2nd_quant,  E_FCI ) 
table_reduced_2nd_quant   =   np.append(table_reduced_2nd_quant, E_FCI) 

print("==================================================================================");
#---LiH (10-quibits)
print("     10-qubits: Full Hamiltonian, 2-frozen core MSO, 8 (all)  vritual  MSO ");
MSO_occ_tmp         = [0,1,        6,7  ] 
MSO_frozen_list_tmp = [0,          6    ]
MSO_active_list_tmp = [  1,2,3,4,5,  7,8,9,10,11 ]

[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                          MSO_occ_tmp,
                          MSO_frozen_list_tmp,
                          MSO_active_list_tmp,
                          Hamiltonian_0body,Hamiltonian_1body,gmo) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)

MSO_occ_mp2   =  [ item for  item in  MSO_occ_tmp  if item not in MSO_frozen_list_tmp  ]
MSO_virt_mp2  =  [ item for  item in  MSO_active_list_tmp  if item not in MSO_occ_mp2  ]
eps_mp2       =   eps  ;
#eps_mp2       =   eps[MSO_active_list])
E_MP2_corr    =   get_MP2_corr(MSO_occ_mp2,MSO_virt_mp2,eps_mp2,gmo,False)


E_FCI =  FCI_eigenvalues[0]
E_FCI_corr = E_FCI - scf_e
print('Occupied MSO: ',MSO_occ_tmp)
print('Frozen   MSO: ',MSO_frozen_list_tmp)
print('Active   MSO: ',MSO_active_list_tmp)
print('E_HF           = ',scf_e)
print('E_FCI          = ',E_FCI)
print('E_FCI_corr     = ',E_FCI_corr)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print('MP2 correlation energy: ', E_MP2_corr)
print('MP2 total energy:       ', scf_e +E_MP2_corr)
print("  ");

print(" Now use reduced  (frozen core) 2nd-quantization Hamiltonian \n ");

Hamiltonian_fc_0body,Hamiltonian_fc_1body,Hamiltonian_fc_2body= get_frozen_core_Hamiltonian(
                                                              MSO_frozen_list_tmp,
                                                              MSO_active_list_tmp,
                                                              E_nucl, 
                                                              Hamiltonian_1body, 
                                                              gmo) ;
 
MSO_occ_tmp_fc         = [0,  5]                    # <- [0,1,        6,7  ]
MSO_frozen_list_tmp_fc = [ ]                        # <- [0,          6    ] 
MSO_active_list_tmp_fc = [0, 1,2,3,4,5,6, 7,8,9  ]  # <- [  1,2,3,4,5,  7,8,9,10,11 ]

[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                              MSO_occ_tmp_fc,
                              MSO_frozen_list_tmp_fc,
                              MSO_active_list_tmp_fc,
                              Hamiltonian_fc_0body,
                              Hamiltonian_fc_1body,
                              Hamiltonian_fc_2body) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)

E_FCI_fc      =  FCI_eigenvalues[0]
E_FCI_corr_fc = E_FCI_fc - scf_e
print('Occupied MSO (fc): ',MSO_occ_tmp_fc)
print('Frozen   MSO (fc): ',MSO_frozen_list_tmp_fc)
print('Active   MSO (fc): ',MSO_active_list_tmp_fc)
print('E_HF            = ',scf_e)
print('E_FCI      (fc) = ',E_FCI_fc)
print('E_FCI_corr (fc) = ',E_FCI_corr_fc)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print("  ");


table_no_of_qbits         =   np.append(table_no_of_qbits, len(MSO_active_list_tmp)) ; 
table_full_2nd_quant      =   np.append(table_full_2nd_quant,  E_FCI ) 
table_reduced_2nd_quant   =   np.append(table_reduced_2nd_quant, E_FCI_fc) 



print("==================================================================================");
print("   6-qubits: 8 total MSO, 2-frozen  MSO, 2-occupied active, 4- vritual active MSO  ");
#---LiH (6-quibits)
MSO_occ_tmp         = [0,1,        6,7      ] 
MSO_frozen_list_tmp = [0,          6        ]
MSO_active_list_tmp = [  1,2,5,      7,8,11 ]




[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                          MSO_occ_tmp,
                          MSO_frozen_list_tmp,
                          MSO_active_list_tmp,
                          Hamiltonian_0body,Hamiltonian_1body,gmo) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)

MSO_occ_mp2   =  [ item for  item in  MSO_occ_tmp  if item not in MSO_frozen_list_tmp  ]
MSO_virt_mp2  =  [ item for  item in  MSO_active_list_tmp  if item not in MSO_occ_mp2  ]
eps_mp2       =   eps  ;
#eps_mp2       =   eps[MSO_active_list])
E_MP2_corr    =   get_MP2_corr(MSO_occ_mp2,MSO_virt_mp2,eps_mp2,gmo,False)



E_FCI =  FCI_eigenvalues[0]
E_FCI_corr = E_FCI - scf_e
print('Occupied MSO: ',MSO_occ_tmp)
print('Frozen   MSO: ',MSO_frozen_list_tmp)
print('Active   MSO: ',MSO_active_list_tmp)
print('E_HF           = ',scf_e)
print('E_FCI          = ',E_FCI)
print('E_FCI_corr     = ',E_FCI_corr)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print('MP2 correlation energy: ', E_MP2_corr)
print('MP2 total energy:       ', scf_e +E_MP2_corr)
print("  ");

print(" Now use reduced  (frozen core) 2nd-quantization Hamiltonian \n ");

Hamiltonian_fc_0body,Hamiltonian_fc_1body,Hamiltonian_fc_2body= get_frozen_core_Hamiltonian(
                                                              MSO_frozen_list_tmp,
                                                              MSO_active_list_tmp,
                                                              E_nucl, 
                                                              Hamiltonian_1body, 
                                                              gmo) ;
 
MSO_occ_tmp_fc         = [0,  3]          # <- [0,1,        6,7  ]
MSO_frozen_list_tmp_fc = [ ]              # <- [0,          6    ] 
MSO_active_list_tmp_fc = [0, 1,2,3,4,5]   # <- [  1,2,3,4,5,  7,8,9,10,11 ]

[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                              MSO_occ_tmp_fc,
                              MSO_frozen_list_tmp_fc,
                              MSO_active_list_tmp_fc,
                              Hamiltonian_fc_0body,
                              Hamiltonian_fc_1body,
                              Hamiltonian_fc_2body) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)

E_FCI_fc      =  FCI_eigenvalues[0]
E_FCI_corr_fc =  E_FCI - scf_e
print('Occupied MSO (fc): ',MSO_occ_tmp_fc)
print('Frozen   MSO (fc): ',MSO_frozen_list_tmp_fc)
print('Active   MSO (fc): ',MSO_active_list_tmp_fc)
print('E_HF            = ',scf_e)
print('E_FCI      (fc) = ',E_FCI_fc)
print('E_FCI_corr (fc) = ',E_FCI_corr_fc)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print("  ");


table_no_of_qbits         =   np.append(table_no_of_qbits, len(MSO_active_list_tmp)) ; 
table_full_2nd_quant      =   np.append(table_full_2nd_quant,  E_FCI ) 
table_reduced_2nd_quant   =   np.append(table_reduced_2nd_quant, E_FCI_fc) 



print("==================================================================================");
#---LiH (4-quibits)
print("   4-qubits: 6 total MSO, 2-frozen  MSO, 2-occupied active, 2-vritual active MSO  ");
MSO_occ_tmp         = [0,1,        6,7      ] 
MSO_frozen_list_tmp = [0,          6        ]
MSO_active_list_tmp = [  1,    5,    7, 11  ] 



[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                          MSO_occ_tmp,
                          MSO_frozen_list_tmp,
                          MSO_active_list_tmp,
                          Hamiltonian_0body,Hamiltonian_1body,gmo) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)

MSO_occ_mp2   =  [ item for  item in  MSO_occ_tmp  if item not in MSO_frozen_list_tmp  ]
MSO_virt_mp2  =  [ item for  item in  MSO_active_list_tmp  if item not in MSO_occ_mp2  ]
eps_mp2       =   eps  ;
#eps_mp2       =   eps[MSO_active_list])
E_MP2_corr    =   get_MP2_corr(MSO_occ_mp2,MSO_virt_mp2,eps_mp2,gmo,False)


E_FCI =  FCI_eigenvalues[0]
E_FCI_corr = E_FCI - scf_e
print('Occupied MSO: ',MSO_occ_tmp)
print('Frozen   MSO: ',MSO_frozen_list_tmp)
print('Active   MSO: ',MSO_active_list_tmp)
print('E_HF           = ',scf_e)
print('E_FCI          = ',E_FCI)
print('E_FCI_corr     = ',E_FCI_corr)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print('MP2 correlation energy: ', E_MP2_corr)
print('MP2 total energy:       ', scf_e +E_MP2_corr)
print("  ");

print(" Now use reduced  (frozen core) 2nd-quantization Hamiltonian \n ");

Hamiltonian_fc_0body,Hamiltonian_fc_1body,Hamiltonian_fc_2body= get_frozen_core_Hamiltonian(
                                                              MSO_frozen_list_tmp,
                                                              MSO_active_list_tmp,
                                                              E_nucl, 
                                                              Hamiltonian_1body, 
                                                              gmo) ;
 
MSO_occ_tmp_fc         = [0,  2]      # <- [0,1,        6,7  ]
MSO_frozen_list_tmp_fc = [ ]          # <- [0,          6    ] 
MSO_active_list_tmp_fc = [0, 1,2,3]   # <- [  1,2,3,4,5,  7,8,9,10,11 ]

[FCI_list,fci_ham_sparse]=get_FCI_hamiltonian(
                              MSO_occ_tmp_fc,
                              MSO_frozen_list_tmp_fc,
                              MSO_active_list_tmp_fc,
                              Hamiltonian_fc_0body,
                              Hamiltonian_fc_1body,
                              Hamiltonian_fc_2body) 

FCI_eigenvalues, FCI_eigenvectors= scipy.sparse.linalg.eigsh(fci_ham_sparse, k=2)

E_FCI_fc      =  FCI_eigenvalues[0]
E_FCI_corr_fc =  E_FCI - scf_e
print('Occupied MSO (fc): ',MSO_occ_tmp_fc)
print('Frozen   MSO (fc): ',MSO_frozen_list_tmp_fc)
print('Active   MSO (fc): ',MSO_active_list_tmp_fc)
print('E_HF            = ',scf_e)
print('E_FCI      (fc) = ',E_FCI_fc)
print('E_FCI_corr (fc) = ',E_FCI_corr_fc)
print('FCI_list (basis) length             : ',len(FCI_list))
print('#of non-zeros in sparse hamiltonian : ',  fci_ham_sparse.getnnz())
print("  ");

table_no_of_qbits         =   np.append(table_no_of_qbits, len(MSO_active_list_tmp)) ; 
table_full_2nd_quant      =   np.append(table_full_2nd_quant,  E_FCI ) 
table_reduced_2nd_quant   =   np.append(table_reduced_2nd_quant, E_FCI_fc) 



print("==========================================================================");
print("   MP2 energy test   \n\n");
#



#--- LiH (4 qubits)
MSO_frozen_list_tmp  = [ 0,     6 ]
MSO_active_list_tmp  = [   1,5,  7,11]
MSO_occ_mp2  = [ 1,    7   ]   # active occupied
MSO_virt_mp2 = [   5,    11]   # active virtual
eps_mp2       =   np.array(eps ) ;


print("Test if reduced Hamiltonian gives the same MP2 energy as full Hamiltonian")
print("Full space info:")
print('MSO_occ  :',MSO_occ_mp2)
print('MSO_virt :',MSO_virt_mp2)
print("")

E_MP2_corr    =   get_MP2_corr(MSO_occ_mp2,MSO_virt_mp2,eps_mp2,gmo,False)

print('SCF energy:                  ', scf_e)
print('MP2 correlation energy (1) : ', E_MP2_corr)
print('MP2 total energy:      (1) : ', scf_e +E_MP2_corr)
print("")


eps_mp2          = eps[MSO_active_list_tmp]     # eps[ 1,5,  7,11]   ;
MSO_occ_mp2      = [ 0,   2   ] 
MSO_virt_mp2     = [   1,   3 ] 
MSO_active_mp2   = [ 0,1, 2,3 ]

E_MP2_corr = get_MP2_corr(MSO_occ_mp2,MSO_virt_mp2,eps_mp2,Hamiltonian_fc_2body,False)

print('MP2 correlation energy (2) : ', E_MP2_corr)
print('MP2 total energy:      (2) : ', scf_e +E_MP2_corr)
print("")


#-----------------------------------------------


print("\n\n\n");
print("    SUMMARY:  \n" );

print("No of quibits/active MSO        : ", table_no_of_qbits );
print("FCI energies, full       2nd Ham: ", table_full_2nd_quant ) ; 
print("FCI energies, fc reduced 2nd Ham: ", table_reduced_2nd_quant );

