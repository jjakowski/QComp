
import numpy as np

#############################################################################################################
def get_gao(I_ao):
   ''' 
      Returns transformed atomic integrals  from chemical to  antisymmetrized physicist notation 
     (orbitals -> spinorbitals)  where spinorbitals aar spin-blocked  
     input:  
	    I_ao(Nb,Nb,Nb,Nb) =  (p,q| r,s) 

            a.o. integrals in chemical notaion   over atomic orbitals 

     output:
            gao(p',r',q',s') = < pr || qs> =  <pr|qs> - <pr|sq>  = (pq|rs)- (ps|rq)

            a.o. integrals in physical notation over atomic  spinorbitasl 
            in spin blocked form (all alpha, followed by all beta)
   '''
   identity = np.eye(2)
   I = np.kron(identity, I_ao)
                                             #  Chemical notation ints:
   I_spinblock = np.kron(identity, I.T)      #      I_spinblock(p,q,r,s) =  (pq | rs ) 
                                             #  Physicists' notation ints:
   tmp = I_spinblock.transpose(0, 2, 1, 3)   #      tmp(p,r,q,s)= <pr|qs>    :    (pq | rs) --->   <pr|qs>     
                                             #  Antisymmetrize:
   gao = tmp - tmp.transpose(0, 1, 3, 2)     #      gao(p,r,q,s)  = <pr||qs>   =  <pr | qs> - <pr | sq>
   return gao 
#############################################################################################################


################################################################################################ 
def get_Hamiltonian_1body(Hn,C):         #2nd_quantization(Enucl,  Hcore, gao,  C) 
    '''
         Generate  second quantization Hamiltonian from   Enucl,  Hcore, gao in a.o. basis set, 
          C  ia assumed to be be of '(2*Nb)x (2*Nb' 
    '''
    Nb_H =  Hn.shape[0]  
    Nb_C =  C.shape[0] 
    if (Nb_C ==  2*Nb_H):  
       # C   runs over spin-orbitals, while H_core  over orbitals,needs to spin-block  H_core
       Hamiltonian_1body_ao = np.block([[Hn               , np.zeros_like(Hn)],
                                        [np.zeros_like(Hn), Hn               ]])
       Hamiltonian_1body    =  np.einsum('ij, jk, kl -> il', C.T,Hamiltonian_1body_ao, C) 
    elif (Nb_C ==  Nb_H):
       Hamiltonian_1body_ao =  Hn 
       Hamiltonian_1body    =  np.einsum('ij, jk, kl -> il', C.T,Hamiltonian_1body_ao, C) 
    else:
       print("get_Hamiltonian_1body(H_core_ao,C):   I have no idea what to do. \n")
       print(" Chekc shapes  of your matrices: \n")
       print("H_core_ao.shape  =  :  %d \n", H_core_ao.shape )
       print("        C.shape  =  :  %d \n", C.shape )
       Hamiltonian_1body = 0
    return  Hamiltonian_1body

################################################################################################ 
def   get_Hamiltonian_2body(gao,C):
   '''
      Transform gao, which is the spin-blocked 4d array of physicist's notation, 
      antisymmetric two-electron integrals, into the MO basis using MO coefficients 
   '''
   gmo = np.einsum('pQRS, pP -> PQRS',
         np.einsum('pqRS, qQ -> pQRS',
         np.einsum('pqrS, rR -> pqRS',
         np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)
   #-- you can save gmo into file or print it:
   #np.savetxt("gmo.txt",gmo.reshape(nso*nso*nso*nso))
   # print("gmo:",gmo.reshape(nso*nso*nso*nso)) 
   #-- to read in octave use this:
   # V=load("gmo.txt")  
   # gmo=  reshape(V, nso,nso,nso,nso) 
   return  gmo
################################################################################################ 
def  get_spin_blocked(A):
   '''
      Duplicate matrix A into spin spin block version: 
           [[   A(alpha) ,  0        ]
            [   0        ,  A(beta)  ]] ;
   '''   
   A_spin  = np.block([[             A   , np.zeros_like(A)],
                       [np.zeros_like(A) ,               A ]])
   return  A_spin  
################################################################################################ 


#----------------
def get_FCI_hamiltonian(MSO_occ,MSO_frozen_list,MSO_active_list,Hamiltonian_0body,Hamiltonian_1body,gmo):
    ''' 
     input:  MSO_occ, MSO_frozen_list, MSO_active_list, 
            Hamiltonian_0body, Hamiltonian_1body, gmo
            where gmo(p,q,r,s) =  <p,q||r,s>     
     output: FCI_list, fci_ham_sparse        

    '''
    from itertools import combinations  
    from scipy.sparse import coo_matrix

    #--  check  if each OCCUPIED MSO is declared ACTVIVE or FROZEN  ---#
    MSO_test =[item for item in MSO_occ if item not in MSO_frozen_list and  item not in MSO_active_list]  
    if  (len(MSO_test) >0):   
       print("");
       print("******** ERROR in MSO list. ********")
       print("Occupied MSO should appear on frozen or active list")
       print("Occupied MSO: ",MSO_occ);
       print("Frozen   MSO: ",MSO_frozen_list);
       print("Active   MSO: ",MSO_active_list);
       print("Dont know what to do with these MSO:" ,MSO_test) 
       print("Exiting.");
       print();
       print("************************************");
       return ;
       #exit()

    #--  check  if each OCCUPIED MSO is declared ACTVIVE or FROZEN  ---#
    #MSO_test =[item for item in MSO_occ if item not in MSO_frozen_list and  item not in MSO_active_list]  
    MSO_test =[item for item in MSO_frozen_list if item in  MSO_active_list] 
    if  (len(MSO_test) >0):   
       print("");
       print("******** ERROR in MSO list. ********")
       print("Some MSO appear on both  ACTIVE and FROZEN list  at the same.")
       print("Occupied MSO: ",MSO_occ);
       print("Frozen   MSO: ",MSO_frozen_list);
       print("Active   MSO: ",MSO_active_list);
       print("Dont know what to do with these MSO:" ,MSO_test) 
       print("Exiting.");
       print();
       print("************************************");
       return ;
       #exit()
        

    #
    N_active_electrons =   len(MSO_occ)-len(MSO_frozen_list)
    FCI_list = [list(x) for x in combinations(MSO_active_list, N_active_electrons) ]
    n_dets= len(FCI_list)    
    #fci_ham =np.zeros((n_dets,n_dets))

    #
    row          = np.array([])
    col          = np.array([])
    fci_ham_data = np.array([])
    #fci_ham_sparse =  coo_matrix((n_dets, n_dets), dtype=np.float64)
    #
    #--- diagonal contributions from frozen  core:
    E_const  =  Hamiltonian_0body
    E_const += sum([Hamiltonian_1body[p,p] for  p in  MSO_frozen_list ])
    #E_const += 0.5*sum([Hamiltonian_2body[p,q,p,q] for p in  MSO_frozen_list for q in MSO_frozen_list])
    E_const += 0.5*sum([gmo[p,q,p,q] for p in  MSO_frozen_list for q in MSO_frozen_list])
    #print(Hamiltonian_2body)
    iact = 0 
    thrs  = 1e-10   #  threshold for sparse elements
    #
    for iDet0 in range(len(FCI_list)): 
        Det0 = FCI_list[iDet0]    
        for iDet1 in range(len(FCI_list)): 
            Det1 = FCI_list[iDet1]
            #r,s= FCI_list[iDet1]
            diff0=[item for item in Det0 if item not in Det1]
            diff1=[item for item in Det1 if item not in Det0]
            same =[item for item in Det0 if item  in Det1] 
            #print( iDet0,iDet1,Det0,Det1,diff0, diff1,'same=',same) #Det0.index(diff))
            if   (len(diff0)==2 and len(diff1)==2): 
                [p,q] = diff0 
                [r,s] = diff1
                #print('gmo:',p,q,r,s,gmo[p,q,r,s])
                #print('indx',Det0.index(p),Det0.index(q),Det1.index(r),Det1.index(s))
                mysign = (-1)**(Det0.index(p)+Det0.index(q)+Det1.index(r)+Det1.index(s))
                fci_tmp =  gmo[p,q,r,s]*mysign
                #fci_ham[iDet0,iDet1] = gmo[p,q,r,s]*mysign               
            elif (len(diff0)==1 and len(diff1)==1):
                [p] = diff0 
                [q] = diff1 
                mysign = (-1)**(Det0.index(p)+Det1.index(q))
                #print(":::", Det0,Det1,diff0, diff1,'same=',same, 'H + gmo:',p,q) #Det0.index(diff))
                H1  =  Hamiltonian_1body[p,q]   
                H1 +=  sum([gmo[r,p,r,q]   for r in MSO_frozen_list ])
                H2  =  sum([gmo[r,p,r,q]   for r in same ])
                #fci_ham[iDet0,iDet1] = (H1 +  H2)*mysign
                fci_tmp =  (H1 +  H2)*mysign
            elif  (len(diff0)==0 and len(diff1)==0):
                H1  = sum([Hamiltonian_1body[p,p] for  p in  same ])
                #H1 += sum([Hamiltonian_2body[p,q,p,q] for  p in  MSO_frozen_list  for q in same])
                #H2  = 0.5*sum([Hamiltonian_2body[p,q,p,q] for  p in  same  for q in same])
                H1 +=     sum([gmo[p,q,p,q] for  p in  MSO_frozen_list  for q in same])
                H2  = 0.5*sum([gmo[p,q,p,q] for  p in  same             for q in same])                
                #H2  = 0.5*sum([gmo[p,q,p,q] for  p in  same  for q in same])
                #H2  = 0.5*sum(Hamiltonian_2body[p,q,p,q] for  p in  same  for q in same])
                #fci_ham[iDet0,iDet1] =  E_const + H1 + H2   
                fci_tmp =     E_const + H1 + H2
            else:
                #fci_ham[iDet0,iDet1] = 0
                fci_tmp = 0 
                #print('H=0')
            #if  abs(fci_ham[iDet0,iDet1]) >  thrs  :
            if  abs(fci_tmp)  >  thrs  :    
                row =  np.append(row,iDet0)
                col =  np.append(col,iDet1)
                fci_ham_data = np.append(fci_ham_data,fci_tmp)
    fci_ham_sparse =  coo_matrix((fci_ham_data,(row,col)),shape=(n_dets, n_dets),dtype=np.float64)
    fci_ham_sparse =  fci_ham_sparse.tocsr()
    #    return  FCI_list, fci_ham, fci_ham_sparse    
    return  FCI_list, fci_ham_sparse

#help(get_FCI_hamiltonian)



##########################################
#----------------
def get_CI_hamiltonian(MSO_occ,MSO_frozen_list,MSO_active_list,Hamiltonian_0body,Hamiltonian_1body,gmo,FCISD):
    ''' 

     input:  MSO_occ, MSO_frozen_list, MSO_active_list, 
            Hamiltonian_0body, Hamiltonian_1body, gmo   
            FCSID:  0=diagonal (HF states),   1=(CIS) , 2=(CID) ,  3=CISD,  10=FCI
     output: FCI_list, fci_ham_sparse        

    '''
    from itertools import combinations  
    from scipy.sparse import coo_matrix
    CI0= 0;   CIS=1;  CID=2  ; CISD=3  ; FCI =  10;
    #
    #
    mso_active_occ_list  = [ x for  x in  MSO_active_list  if x       in MSO_occ  ]     # MSO_occ =  reference HF state
    mso_active_virt_list = [ x for  x in  MSO_active_list  if x  not  in MSO_occ  ]     # 
    print("mso_frozen_list :\n",     MSO_frozen_list )
    print("mso_active_occ_list :\n", mso_active_occ_list)
    print("mso_active_virt_list :\n",mso_active_virt_list)
    #
    # generate  list oc determinants
    # 
    if  (FCISD ==  FCI):      # full-CI case
        N_active_electrons =   len(MSO_occ)-len(MSO_frozen_list)
        CI_list = [sorted(list(x)) for x in combinations(MSO_active_list, N_active_electrons) ]
        n_dets= len(CI_list)  
    elif (FCISD ==  CIS):       # CI-S case 
        Nexc =  1; 
        Nel =   len(MSO_occ)-len(MSO_frozen_list) 
        CIS_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  for y in  combinations(mso_active_virt_list, Nexc)]
        CI_list=  [ MSO_occ ] + CIS_list                 # = reference_HF + single_excitations
    elif (FCISD ==  CID):       # CI-D case  
        Nexc =  2; 
        Nel =   len(MSO_occ)-len(MSO_frozen_list) 
        CID_list = [ sorted(MSO_frozen_list +list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  for y in  combinations(mso_active_virt_list, Nexc)]
        CI_list=   [ MSO_occ ] +  CID_list               # = reference_HF + double_excitations
    elif (FCISD ==  CISD):      # CI-SD  case
        Nel =   len(MSO_occ)-len(MSO_frozen_list) 
        Nexc =  1; 
        CIS_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  for y in  combinations(mso_active_virt_list, Nexc)]
        Nexc =  2; 
        CID_list = [ sorted(MSO_frozen_list +list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  for y in  combinations(mso_active_virt_list, Nexc)]
        CI_list= [ MSO_occ ] + CIS_list  + CID_list      #  = reference_HF + single_excitations + double_excitations
    elif  (FCISD == CI0):          # HF -type states (diagonal elements in CIS)
        Nexc =  1; 
        Nel =   len(MSO_occ)-len(MSO_frozen_list) 
        #CIS_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  for y in  combinations(mso_active_virt_list, Nexc)]
        #CI_list=  [ MSO_occ ] + CIS_list                 # = reference_HF + single_excitations    
        CI_list =  [ MSO_occ ]
    else:  
        print("get_CI_hamiltonian: I dont know what to do ? \n")
        CI_list=[] 
    print("CIS_list :  \n",CI_list)
    print("len(CI_list) = ", len(CI_list) )
    #
    #
    n_dets= len(CI_list)    
    #ci_ham =np.zeros((n_dets,n_dets))
    #
    row         = np.array([])
    col         = np.array([])
    ci_ham_data = np.array([])
    #ci_ham_sparse =  coo_matrix((n_dets, n_dets), dtype=np.float64)
    #
    #--- diagonal contributions from frozen  core:
    E_const  =  Hamiltonian_0body
    E_const += sum([Hamiltonian_1body[p,p] for  p in  MSO_frozen_list ])
    E_const += 0.5*sum([gmo[p,q,p,q] for p in  MSO_frozen_list for q in MSO_frozen_list])
    iact = 0 
    thrs  = 1e-10   #  threshold for sparse elements
    #
    for iDet0 in range(len(CI_list)): 
        Det0 = CI_list[iDet0]    
        for iDet1 in range(len(CI_list)): 
            Det1 = CI_list[iDet1]
            #r,s= CI_list[iDet1]
            diff0=[item for item in Det0 if item not in Det1]
            diff1=[item for item in Det1 if item not in Det0]
            same =[item for item in Det0 if item  in Det1] 
            #print( iDet0,iDet1,Det0,Det1,diff0, diff1,'same=',same) #Det0.index(diff))
            if   (len(diff0)==2 and len(diff1)==2): 
                [p,q] = diff0 
                [r,s] = diff1
                #print('gmo:',p,q,r,s,gmo[p,q,r,s])
                #print('indx',Det0.index(p),Det0.index(q),Det1.index(r),Det1.index(s))
                mysign = (-1)**(Det0.index(p)+Det0.index(q)+Det1.index(r)+Det1.index(s))
                ci_tmp =  gmo[p,q,r,s]*mysign
                #ci_ham[iDet0,iDet1] = gmo[p,q,r,s]*mysign               
            elif (len(diff0)==1 and len(diff1)==1):
                [p] = diff0 
                [q] = diff1 
                mysign = (-1)**(Det0.index(p)+Det1.index(q))
                #print(":::", Det0,Det1,diff0, diff1,'same=',same, 'H + gmo:',p,q) #Det0.index(diff))
                H1  =  Hamiltonian_1body[p,q]   
                H1 +=  sum([gmo[r,p,r,q]   for r in MSO_frozen_list ])
                H2  =  sum([gmo[r,p,r,q]   for r in same ])
                #ci_ham[iDet0,iDet1] = (H1 +  H2)*mysign
                ci_tmp =  (H1 +  H2)*mysign
            elif  (len(diff0)==0 and len(diff1)==0):
                H1  = sum([Hamiltonian_1body[p,p] for  p in  same ])
                #H1 += sum([Hamiltonian_2body[p,q,p,q] for  p in  MSO_frozen_list  for q in same])
                #H2  = 0.5*sum([Hamiltonian_2body[p,q,p,q] for  p in  same  for q in same])
                H1 +=     sum([gmo[p,q,p,q] for  p in  MSO_frozen_list  for q in same])
                H2  = 0.5*sum([gmo[p,q,p,q] for  p in  same             for q in same])                
                #H2  = 0.5*sum([gmo[p,q,p,q] for  p in  same  for q in same])
                #H2  = 0.5*sum(Hamiltonian_2body[p,q,p,q] for  p in  same  for q in same])
                #ci_ham[iDet0,iDet1] =  E_const + H1 + H2   
                ci_tmp =     E_const + H1 + H2
            else:
                #ci_ham[iDet0,iDet1] = 0
                ci_tmp = 0 
                #print('H=0')
            #if  abs(ci_ham[iDet0,iDet1]) >  thrs  :
            if  abs(ci_tmp)  >  thrs  :    
                row =  np.append(row,iDet0)
                col =  np.append(col,iDet1)
                ci_ham_data = np.append(ci_ham_data,ci_tmp)
    ci_ham_sparse =  coo_matrix((ci_ham_data,(row,col)),shape=(n_dets, n_dets),dtype=np.float64)
    ci_ham_sparse =  ci_ham_sparse.tocsr()
    return  CI_list, ci_ham_sparse




###############################################

#----
def get_MP2_corr(MSO_occ,MSO_virt,eps,gmo,ifprint_amps):
    '''
     Calculate MP2 energy and amplitudes. Arguments:
    
     MSO_occ     : 1D list of indices for occupied molecular spin-orbitals used for MP2  calcs 
     MSO_virt    : 1D list of indices for virtual molecular spin-orbitals used for MP2 calcs
                   Notice, that together MSO_occ and MSO_virt are  active active orbitals  
                   which can a subset of all orbitals. 
     eps         : 1D list containig energies ALL molecular spin-orbital
     gmo         : 4D tensor of  integrals  <pq||rs> over ALL spinorbitals
     ifprint_amps:   This should be set to 'True'  or 'False'  depending on whether 
                  printing of   MP2 amplitudes is requesed or not
    '''
    E_MP2_corr=  0
    n_occ  = len(MSO_occ )
    n_virt = len(MSO_virt)
    for  i in range(0,n_occ,1):
        ii  = MSO_occ[i];  
        e_i = eps[ii]
        for j in range(0,i,1):
            jj  = MSO_occ[j];
            e_j = eps[jj]
            for  a in range(0,n_virt,1):
                ia  = MSO_virt[a]
                e_a = eps[ia]
                for  b in range(0,a,1):
                    ib  = MSO_virt[b]
                    e_b = eps[ib]
                    de_ijab = (e_i+e_j -e_a -e_b)  
                    t_ijab =  gmo[ii,jj,ia,ib]/de_ijab      # MP2 amplitudes 
                    E_MP2_corr += t_ijab*gmo[ii,jj,ia,ib]       # MP2 corrections
                    if (ifprint_amps == True): print(ii,jj,'->',ia,ib,':',  t_ijab)
    #print('MP2 correlation energy: ', E_MP2_corr)
    return  E_MP2_corr
##################################################3

def   get_frozen_core_Hamiltonian( MSO_frozen_list,
                                   MSO_active_list,
                                   Hamiltonian_0body, 
                                   Hamiltonian_1body, 
                                   Hamiltonian_2body):
    '''
      Get   frozen core hamiltonian with second quantization 

      input: Hamiltonian_0body, Hamiltonian_1body, Hamiltonian_2body,
             MSO_frozen_list,MSO_active_list 

      output: Hamiltonian_fc_0body, Hamiltonian_ifc_1body, Hamiltonian_fc_2body
         MSO_frozen_list:  1D list of indices for  frozen, occupied molecular spin-orbitials 
        MSO_active_list :  1D list of indices  of active orbitals (occupied and virtual)

    '''
    #- 
    n_frozen = len(MSO_frozen_list)
    n_active = len(MSO_active_list)

    gmo =  Hamiltonian_2body
    
    #----- 0-body frozen-core:
    Hamiltonian_fc_0body = Hamiltonian_0body  ; # E_nucl 
    for a in range(n_frozen):
        ia = MSO_frozen_list[a]
        Hamiltonian_fc_0body += Hamiltonian_1body[ia,ia]
        for b in range(a):
            ib = MSO_frozen_list[b]
            Hamiltonian_fc_0body += gmo[ia,ib,ia,ib]

        
    #--- 1-body frozen-core:
    Hamiltonian_fc_1body =np.zeros((n_active,n_active))
    Hamiltonian_fc_1body_tmp =np.zeros((n_active,n_active))
    for p in range(n_active):
        ip = MSO_active_list[p]
        for  q in range(n_active):
            iq =  MSO_active_list[q]
            Hamiltonian_fc_1body[p,q] =  Hamiltonian_1body[ip,iq]
            #Hamiltonian_fc_1body_tmp[p,q] =  Hamiltonian_1body[ip,iq]
            for a in range(n_frozen):
                ia = MSO_frozen_list[a]
                Hamiltonian_fc_1body[p,q] += gmo[ia,ip,ia,iq]

    #------- 2-body frozen-core:
    Hamiltonian_fc_2body =np.zeros((n_active,n_active,n_active,n_active))
    for p in range(n_active):
        ip = MSO_active_list[p]
        for q in range(n_active):
            iq = MSO_active_list[q]
            for r in range(n_active):
                ir = MSO_active_list[r]
                for ss in range(n_active):
                    iss =  MSO_active_list[ss]
                    #Hamiltonian_fc_2body[p,q,r,ss]= 0.25* gmo[ip,iq,ir,iss]
                    Hamiltonian_fc_2body[p,q,r,ss]=  gmo[ip,iq,ir,iss]
                    #Hamiltonian_fc_2body[p,q,r,ss]= 0.25* gmo[ip,iq,iss,ir]
    return Hamiltonian_fc_0body,Hamiltonian_fc_1body,Hamiltonian_fc_2body

####################################################################################
def  generate_CI_list(MSO_occ,MSO_frozen_list,MSO_active_list,CI_list_type):
    '''         
     Generate list of determinants for CI.

     input:  MSO_occ           - list of occupied Molecular Spin Orbitals (MSO)
                                   MSO_occ is a reference determinant. 
             MSO_frozen_list   - list of frozen MSO
             MSO_active_list   - list of active MSO 
             CI_list_type      -  0  for HF, 1 for CIS, 2 for CID, 3 for CISD etc.
             
            Note:  each MSO in MSO_occ should be marked as frozen or active
            
     output: CI_list   -  list of Slater determinants 

    '''
    from itertools import combinations
    from scipy.sparse import coo_matrix

    #--  check  if each OCCUPIED MSO is declared ACTVIVE or FROZEN  ---#
    MSO_test =[item for item in MSO_occ if item not in MSO_frozen_list and  item not in MSO_active_list]
    if  (len(MSO_test) >0):
       print("");
       print("******** ERROR in generate_CI_list(): MSO list.  ********")
       print("Occupied MSO should appear on frozen or active list")
       print("Occupied MSO: ",MSO_occ);
       print("Frozen   MSO: ",MSO_frozen_list);
       print("Active   MSO: ",MSO_active_list);
       print("Dont know what to do with these MSO:" ,MSO_test)
       print("Exiting.");
       print();
       print("*********************************************************");
       #exit()
       return 

    #--  check  if each OCCUPIED MSO is declared ACTVIVE or FROZEN  ---#
    #MSO_test =[item for item in MSO_occ if item not in MSO_frozen_list and  item not in MSO_active_list]  
    MSO_test =[item for item in MSO_frozen_list if item in  MSO_active_list]
    if  (len(MSO_test) >0):
       print("");
       print("******** ERROR in generate_CI_list(): MSO list. ********")
       print("Some MSO appear on both  ACTIVE and FROZEN list  at the same.")
       print("Occupied MSO: ",MSO_occ);
       print("Frozen   MSO: ",MSO_frozen_list);
       print("Active   MSO: ",MSO_active_list);
       print("Dont know what to do with these MSO:" ,MSO_test)
       print("Exiting.");
       print();
       print("*******************************************************");
       #exit()
       return 

    
    #N_active_electrons =   len(MSO_occ)-len(MSO_frozen_list)
    #FCI_list = [list(x) for x in combinations(MSO_active_list, N_active_electrons) ]
    #n_dets= len(FCI_list )
    # Note: MSO_occ is a reference determinant 
    mso_active_occ_list  = [ x for  x in  MSO_active_list  if x       in MSO_occ  ]     
    mso_active_virt_list = [ x for  x in  MSO_active_list  if x  not  in MSO_occ  ]     
    print("-----------------------------------");
    print("mso_frozen_list      :  ",     MSO_frozen_list )
    print("mso_active_occ_list  :  ", mso_active_occ_list)
    print("mso_active_virt_list :  ",mso_active_virt_list)
    #binary       421
    HF = 0 ;  CIS = 1;  CID=2 ;CISD=3 ;   FCI=10;
     
    if (CI_list_type == HF ): 
        #N_active_electrons =   len(MSO_occ)-len(MSO_frozen_list)
        #CI_list = [sorted(list(x)) for x in combinations(MSO_active_list, N_active_electrons)]
        #n_dets= len(CI_list)

        Nexc =  0;
        Nel =   len(MSO_occ)-len(MSO_frozen_list)
        #CI_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )
        #                                                for y in  combinations(mso_active_virt_list, Nexc    )]
        #CI_list=  [ MSO_occ ] + CIS_list                 # = reference_HF + single_excitations
        CI_list  =   MSO_occ                              # = reference_HF 
        #print(CI_list)
    elif (CI_list_type ==  CIS):       # CI-S case 
        Nexc =  1;
        Nel =   len(MSO_occ)-len(MSO_frozen_list)
        CIS_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )
                                                         for y in  combinations(mso_active_virt_list, Nexc    )]
        #CI_list=  [ MSO_occ ] + CIS_list                # = reference_HF + single_excitations
        CI_list =                CIS_list                # =                single_excitations
        #print(CIS_list)
    elif (CI_list_type ==  CID):       # CI-D case  
        Nexc =  2;
        Nel =   len(MSO_occ)-len(MSO_frozen_list)
        CID_list = [ sorted(MSO_frozen_list +list(x+y))  for x in  combinations(mso_active_occ_list, Nel-Nexc )  
                                                         for y in  combinations(mso_active_virt_list, Nexc    )]
        #CI_list=  [ MSO_occ ] +  CID_list               # = reference_HF + double_excitations
        CI_list=                  CID_list               # =                double_excitations
    elif (CI_list_type ==  CISD):      # CI-SD  case
        Nel =   len(MSO_occ)-len(MSO_frozen_list)
        Nexc =  1;
        CIS_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  
                                                         for y in  combinations(mso_active_virt_list, Nexc    )]
        Nexc =  2;
        CID_list = [ sorted(MSO_frozen_list +list(x+y))  for x in  combinations(mso_active_occ_list, Nel-Nexc )  
                                                         for y in  combinations(mso_active_virt_list, Nexc    )]
        #CI_list= [ MSO_occ ] + CIS_list  + CID_list     #  = reference_HF + single_excitations + double_excitations
        CI_list=                CIS_list  + CID_list     #  =                single_excitations + double_excitations
    elif  (CI_list_type == CI0):          # HF -type states (diagonal elements in CIS)
        Nexc =  1;
        Nel =   len(MSO_occ)-len(MSO_frozen_list)
        #CIS_list = [ sorted(MSO_frozen_list+list(x+y))   for x in  combinations(mso_active_occ_list, Nel-Nexc )  for y in  combinations(mso_active_virt_list, Nexc)]
        #CI_list=  [ MSO_occ ] + CIS_list                 # = reference_HF + single_excitations    
        CI_list =  [ MSO_occ ]
    else:
        print("get_CI_list: I dont know what to do ? \n")
        CI_list=[]
    print("CI_list :  \n",CI_list)
    print("len(CI_list) = ", len(CI_list) )
    return CI_list 

##################################################################################################
def  generate_CI_singlet_list(MSO_occ,MSO_frozen_list,MSO_active_list,CI_list_type, Nbasis):
    '''         
     Generate list of determinants for CI with Singlet excitations only.

     input:  MSO_occ           - list of occupied Molecular Spin Orbitals (MSO)
                                   MSO_occ is a reference determinant. 
             MSO_frozen_list   - list of frozen MSO
             MSO_active_list   - list of active MSO 
             CI_list_type      -  0  for HF, 1 for CIS, 2 for CID, 3 for CISD etc.
             
             Nbasis            - Number of  basis functions (=1/2 or MSO)
            Note:  each MSO in MSO_occ should be marked as frozen or active
            
     output: CI_list   -  list of Slater determinants 

    '''
    from itertools import combinations
    from scipy.sparse import coo_matrix

    MSO_occ_alpha =   [ x for  x in  MSO_occ if x   <   Nbasis   ] 
    MSO_occ_beta  =   [ x for  x in  MSO_occ if x   >=  Nbasis   ]
    MSO_frozen_list_alpha  =  [ x for  x in  MSO_frozen_list  if x   <   Nbasis   ]  
    MSO_frozen_list_beta   =  [ x for  x in  MSO_frozen_list  if x   >=  Nbasis   ]  
    MSO_active_list_alpha  =  [ x for  x in  MSO_active_list  if x   <   Nbasis   ]  
    MSO_active_list_beta   =  [ x for  x in  MSO_active_list  if x   >=  Nbasis   ]  
 
    Nel_alpha  =   len(MSO_occ_alpha)-len(MSO_frozen_list_alpha)
    Nel_beta   =   len(MSO_occ_beta) -len(MSO_frozen_list_beta)

    HF   = 0; 
    CIS  = 1;  
    CID  = 2;
    CISD = 3; 

    # prepare half lists  for excitations ( alpha ->alpha) and (beta-> beta)
    if (CI_list_type >= CIS) :
       CIS_half_list_alpha  = generate_CI_list(MSO_occ_alpha, 
                                               MSO_frozen_list_alpha,
                                               MSO_active_list_alpha,  CIS )
       CIS_half_list_beta   = generate_CI_list(MSO_occ_beta, 
                                               MSO_frozen_list_beta,
                                               MSO_active_list_beta,   CIS )

    if (CI_list_type >= CID and Nel_alpha >= 2 ) :    # make sure we hav at least 2-alpha electrons to excite
       CID_half_list_alpha  = generate_CI_list(MSO_occ_alpha, 
                                               MSO_frozen_list_alpha,
                                               MSO_active_list_alpha,  CID )
    if (CI_list_type >= CID and Nel_beta  >= 2 ) :    # make sure we ahve at least 2-beta electrons to excite 
       CID_half_list_beta   = generate_CI_list(MSO_occ_beta, 
                                               MSO_frozen_list_beta,
                                               MSO_active_list_beta,   CID )
           


    if  (CI_list_type == CIS ) :
        # alpha excitations 
        CIS_list_alpha       = [ x + MSO_occ_beta   for x in CIS_half_list_alpha ]
        # abeta excitations 
        CIS_list_beta        = [ MSO_occ_alpha + x  for x in CIS_half_list_beta  ]
        CI_list =  CIS_list_alpha  +  CIS_list_beta 

    elif (CI_list_type == CID) :
        # alpha,beta  --> alpha, beta  excitations
        CID_list_ab = [sorted(x +y) for x in  CIS_half_list_alpha    
                                    for y in  CIS_half_list_beta   ] 
        # alpha,alpha --> alpha,alpha  excitations
        CID_list_aa = [ sorted(x + MSO_occ_beta)   for x in CID_half_list_alpha ]    
        # beta,beta   --> beta,beta excitations
        CID_list_bb = [ sorted(x + MSO_occ_alpha)  for x in CID_half_list_beta  ]    
        CI_list     = CID_list_aa +  CID_list_bb  +CID_list_ab

        print(" CIS_half_list_alpha : ", CIS_half_list_alpha)
        print(" CIS_half_list_beta  : ", CIS_half_list_beta )
        print(" CID_list_aa : ",CID_list_aa)
        print(" CID_list_bb : ",CID_list_bb)
        print(" CID_list_ab : ",CID_list_ab)
    elif (CI_list_type == CISD) :
        # single  alpha excitations 
        CIS_list_alpha       = [ sorted(x + MSO_occ_beta)   for x in CIS_half_list_alpha ]
        # single abeta excitations 
        CIS_list_beta        = [ sorted(x + MSO_occ_alpha)  for x in CIS_half_list_beta  ]

        # double  (alpha,beta)  --> (alpha, beta)  excitations
        CID_list_ab = [ sorted(x +y) for x in  CIS_half_list_alpha    
                                     for y in  CIS_half_list_beta   ] 
        # double (alpha,alpha) --> (alpha,alpha)  excitations
        CID_list_aa = [ sorted(x + MSO_occ_beta)   for x in CID_half_list_alpha ]    
        # double (beta,beta)   --> (beta,beta)  excitations
        CID_list_bb = [ sorted(x + MSO_occ_alpha)  for x in CID_half_list_beta  ]    

        CI_list =CIS_list_alpha  + CIS_list_beta  + CID_list_aa +  CID_list_ab  +CID_list_bb
       
        
    return CI_list 
    #return CI_list_alpha ,  CI_list_beta  
    #return  CID_half_list_alpha,  CID_half_list_beta
    #return  CID_list_ab,  CID_list_aa,  CID_list_bb

##################################################################################################







############################################3
###
#### Utilities
###     

def printMatrix(a,thrs):  
   print("Matrix\n");
   rows = a.shape[0];
   cols = a.shape[1]; 
   for i in range(0,rows): 
      for j in range(0,cols):
         print("%6.f \n" %a[i,j]) ;
      print; 
   print ;

def  print_matrix(A_matrix,thrs) : 
   icount=0 ;
   rows = A_matrix.shape[0];
   cols = A_matrix.shape[1]; 
   for ix in range(0,rows): 
      for iy  in range(0,cols):
         #print("%6.f \n" %a[ix,iy]) ;
         if  abs(A_matrix[ix,iy]) >  thrs   : 
              print("%5d  %5d    %20.15f  " % (ix, iy,A_matrix[ix,iy]))
              icount += 1 ; 
   print(" Total  %d  non-zero elements " % icount);


def print_square(A_matrix): 
    rows = A_matrix.shape[0];
    cols = A_matrix.shape[1];
    for ix in range(0,rows):
        print("") ;
        for iy  in range(0,cols):
             #print("%14.8e " %A_matrix[ix,iy],end='') ;
             #print("%14.10f " %A_matrix[ix,iy],end='') ;
             print("{:14.10f} ".format(A_matrix[ix,iy]),end='') ;
    print("\n")

