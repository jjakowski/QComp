
import psi4
import numpy as np

#import myutils



#--- geometry

HH_geometry = psi4.geometry ("""
0 1
H   0.000000   0.0      0.0 
H   0.8        0.0      0.0 
H   2.000000   0.0      0.0 
H   2.8        0.0      0.0 
H   4.000000   0.0      0.0 
H   4.8        0.0      0.0 
H   6.000000   0.0      0.0 
H   6.8        0.0      0.0 
symmetry c1
no_reorient
no_com
""")


HH_a_geometry = psi4.geometry ("""
0 1
H   0.000000   0.0      0.0 
H   0.8        0.0      0.0 
symmetry c1
no_reorient
no_com
""")

HH_b_geometry = psi4.geometry ("""
0 1
H   2.000000   0.0      0.0 
H   2.8        0.0      0.0 
symmetry c1
no_reorient
no_com
""")

HH_c_geometry = psi4.geometry ("""
0 1
H   4.000000   0.0      0.0 
H   4.8        0.0      0.0 
symmetry c1
no_reorient
no_com
""")

HH_d_geometry = psi4.geometry ("""
0 1
H   6.000000   0.0      0.0 
H   6.8        0.0      0.0 
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
scf_e_a, scf_wfn_a = psi4.energy('scf', return_wfn=True,molecule=HH_a_geometry)
scf_e_b, scf_wfn_b = psi4.energy('scf', return_wfn=True,molecule=HH_b_geometry)
scf_e_c, scf_wfn_c = psi4.energy('scf', return_wfn=True,molecule=HH_c_geometry)
scf_e_d, scf_wfn_d = psi4.energy('scf', return_wfn=True,molecule=HH_d_geometry)

print("E(SCF)_a:", scf_e_a, scf_wfn_a)
print("E(SCF)_b:", scf_e_b, scf_wfn_b)
print("E(SCF)_c:", scf_e_c, scf_wfn_c)
print("E(SCF)_d:", scf_e_d, scf_wfn_d)
print("E(SCF)  :", scf_e  , scf_wfn  )


#nuclear repulsion energy
E_nucl_a = HH_a_geometry.nuclear_repulsion_energy()
E_nucl_b = HH_b_geometry.nuclear_repulsion_energy()
E_nucl_c = HH_c_geometry.nuclear_repulsion_energy()
E_nucl_d = HH_d_geometry.nuclear_repulsion_energy()
E_nucl   = HH_geometry.nuclear_repulsion_energy()

print("Nuclear repulsion (a)  :", E_nucl_a)
print("Nuclear repulsion (b)  :", E_nucl_b)
print("Nuclear repulsion (c)  :", E_nucl_c)
print("Nuclear repulsion (d)  :", E_nucl_d)
print("Nuclear repulsion (all):", E_nucl)

#
# read in matrices from SCF  (a.o.) :
#-- overlaps
S    = np.array(scf_wfn.S())
S_a  = np.array(scf_wfn_a.S())
S_b  = np.array(scf_wfn_b.S())
S_c  = np.array(scf_wfn_c.S())
S_d  = np.array(scf_wfn_d.S())

Fa       =  np.array(scf_wfn.Fa())
Fa_a     =  np.array(scf_wfn_a.Fa())
Fa_a     =  np.array(scf_wfn_b.Fa())
Fa_a     =  np.array(scf_wfn_c.Fa())
Fa_a     =  np.array(scf_wfn_d.Fa())

H        =  np.array(scf_wfn.H())
H_a      =  np.array(scf_wfn_a.H())
H_b      =  np.array(scf_wfn_b.H())
H_c      =  np.array(scf_wfn_c.H())
H_d      =  np.array(scf_wfn_d.H())


Ca_a = np.array(scf_wfn_a.Ca())  ;   Cb_a = np.array(scf_wfn_a.Cb())
Ca_b = np.array(scf_wfn_b.Ca())  ;   Cb_b = np.array(scf_wfn_b.Cb())
Ca_c = np.array(scf_wfn_c.Ca())  ;   Cb_c = np.array(scf_wfn_c.Cb())
Ca_d = np.array(scf_wfn_d.Ca())  ;   Cb_d = np.array(scf_wfn_d.Cb())
Ca   = np.array(scf_wfn.Ca())    ;   Cb   = np.array(scf_wfn.Cb())

#
#---integrals:
# Create instance of MintsHelper class:
mints   = psi4.core.MintsHelper(scf_wfn.basisset())
mints_a = psi4.core.MintsHelper(scf_wfn_a.basisset())
mints_b = psi4.core.MintsHelper(scf_wfn_b.basisset())
mints_c = psi4.core.MintsHelper(scf_wfn_c.basisset())
mints_d = psi4.core.MintsHelper(scf_wfn_d.basisset())

#-- build  a.o matrices
#T = np.asarray(mints.ao_kinetic())
#V = np.asarray(mints.ao_potential())
#H_core_ao = T + V
#mux,muy,muz  = mints.ao_dipole()   #- dipole moment matrix

#-- electron repulsion:
Ints    = np.array(mints.ao_eri())
Ints_a  = np.array(mints_a.ao_eri())
Ints_b  = np.array(mints_b.ao_eri())
Ints_c  = np.array(mints_c.ao_eri())
Ints_d  = np.array(mints_d.ao_eri())


Ia = Ints [0:2,0:2,0:2,0:2]
Ib = Ints [2:4,2:4,2:4,2:4]
Ic = Ints [4:6,4:6,4:6,4:6]
Id = Ints [6:8,6:8,6:8,6:8]

#  lists of basis  id's:
ls_a = [0,1] ;  ls_b = [2,3] ; ls_c = [4,5] ; ls_d =[6,7]
ls_ab = ls_a +ls_b                      #  [0,1] + [2,3]  = [0,1,2,3]   #concatenation
ls_ac = ls_a +ls_c
ls_ad = ls_a +ls_d
ls_bc = ls_b +ls_c
ls_bd = ls_b +ls_d
ls_cd = ls_c +ls_d
 

#--- block integrals
#   Iab  =  Ints [0:4,0:4,0:4,0:4]
Iab = Ints [ls_ab,:,:,:]   [:,ls_ab,:,:]  [:,:,ls_ab,:]  [:,:,:,ls_ab] 
Iac = Ints [ls_ac,:,:,:]   [:,ls_ac,:,:]  [:,:,ls_ac,:]  [:,:,:,ls_ac] 
Iad = Ints [ls_ad,:,:,:]   [:,ls_ad,:,:]  [:,:,ls_ad,:]  [:,:,:,ls_ad] 
Ibc = Ints [ls_bc,:,:,:]   [:,ls_bc,:,:]  [:,:,ls_bc,:]  [:,:,:,ls_bc] 
Ibd = Ints [ls_bd,:,:,:]   [:,ls_bd,:,:]  [:,:,ls_bd,:]  [:,:,:,ls_bd] 
Icd = Ints [ls_cd,:,:,:]   [:,ls_cd,:,:]  [:,:,ls_cd,:]  [:,:,:,ls_cd] 


#--- 
#   Get 2nd  quantization Hamiltonian
#
Hamiltonian_0body_a  =  E_nucl_a ; 
Hamiltonian_0body_b  =  E_nucl_b ; 
Hamiltonian_0body_c  =  E_nucl_c ; 
Hamiltonian_0body_a  =  E_nucl_d ; 
Hamiltonian_0body    =  E_nucl   ; 

Hamiltonian_1body_a =  get_Hamiltonian_1body(H_a, get_spin_blocked(Ca_a))  ;  
Hamiltonian_1body_b =  get_Hamiltonian_1body(H_b, get_spin_blocked(Ca_b))  ;  
Hamiltonian_1body_c =  get_Hamiltonian_1body(H_c, get_spin_blocked(Ca_c))  ;  
Hamiltonian_1body_d =  get_Hamiltonian_1body(H_d, get_spin_blocked(Ca_d))  ;  
Hamiltonian_1body   =  get_Hamiltonian_1body(H  , get_spin_blocked(Ca  ))  ;  

Hamiltonian_2body_a = get_Hamiltonian_2body(get_gao(Ints_a),get_spin_blocked(Ca_a))
Hamiltonian_2body_b = get_Hamiltonian_2body(get_gao(Ints_b),get_spin_blocked(Ca_b))
Hamiltonian_2body_c = get_Hamiltonian_2body(get_gao(Ints_c),get_spin_blocked(Ca_c))
Hamiltonian_2body_d = get_Hamiltonian_2body(get_gao(Ints_d),get_spin_blocked(Ca_d))
Hamiltonian_2body   = get_Hamiltonian_2body(get_gao(Ints  ),get_spin_blocked(Ca  ))

#
#   get  CI hamiltonian
#
from scipy.linalg  import eigh
from scipy.sparse import coo_matrix
import scipy.sparse.linalg 

MSO_occ = [0,2] ;  MSO_frozen_list = [] ;   MSO_active_list = [0,1,2,3] ;

[FCI_list,fci_ham_sparse] =get_FCI_hamiltonian(MSO_occ,MSO_frozen_list,MSO_active_list,Hamiltonian_0body_a, Hamiltonian_1body_a,Hamiltonian_2body_a)



FCI_eigenvalues, FCI_eigenvectors = scipy.sparse.linalg.eigsh(fci_ham_sparse, k=5)
print(FCI_eigenvalues)

# Lowdin transformation of  matrix 
from scipy import linalg, sparse
Sh_a  = linalg.sqrtm(S_a)
Shi_a = linalg.pinv(Sh_a) 
Shi_a2 = linalg.funm(S_a, lambda x:  x**(-1/2))
    
print("Sh_a   =  \n", Sh_a)
print("Shi_a  =  \n", Shi_a)
print("Shi_a2 =  \n", Shi_a2)
  

#array([-4.39538627, -3.98125773, -3.98125773, -3.98125773, -3.88224757])

[Se, Csv] = np.linalg.eig(S_a)

 

#Shi = np.array([  [  1.204709570941634e+00 , -4.192311233095423e-01 ] , [  -4.192311233095423e-01 ,  1.204709570941634]]) 
Shi =  Shi_a ;

Csv =  Shi
H_0body = E_nucl_a
H_1body = get_Hamiltonian_1body(H_a,get_spin_blocked(Csv)) 
H_2body = get_Hamiltonian_2body(get_gao(Ints_a), get_spin_blocked(Csv)) 
[FCI_list_v2,fci_ham_sparse_v2] =get_FCI_hamiltonian(MSO_occ,MSO_frozen_list,MSO_active_list,H_0body,H_1body, H_2body) 

FCI_eigenvalues_v2, FCI_eigenvectors_v2 = scipy.sparse.linalg.eigsh(fci_ham_sparse_v2, k=5)
print('old         : ',FCI_eigenvalues)
print('new         : ',FCI_eigenvalues_v2)







