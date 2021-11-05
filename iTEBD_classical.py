import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
    
def Make_Canonical_form(Tn,lam):
    ## We assume uniforma iMPS
    ## Calculate left and right dominant eigen vectors

    chi = Tn.shape[0]

    Tn_L = np.tensordot(np.diag(lam),Tn,(1,0))
    Tn_R = np.tensordot(Tn,np.diag(lam),(2,0))
    def vL(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn_L,(0,0)),Tn_L.conj(),([0,1],[0,1]))

        return v_new.reshape(chi**2)
    def Rv(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn_R,(0,2)),Tn_R.conj(),([0,2],[2,1]))
        return v_new.reshape(chi**2)


    if chi > 1:    
        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=Rv)
        eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=vL)

        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_R = np.ones(1)
        eig_vec_L = np.ones(1)
        eig_val_R = Rv(eig_vec_R)
        eig_val_L = vL(eig_vec_L)
        
    UR, sR, VTR = linalg.svd(np.real(eig_vec_R).reshape(chi,chi),full_matrices=False)
    UL, sL, VTL = linalg.svd(np.real(eig_vec_L).reshape(chi,chi),full_matrices=False)

    X = np.dot(UR,np.diag(np.sqrt(sR)))
    Y = np.dot(UL,np.diag(np.sqrt(sL)))

    U, lam_new, VT = linalg.svd(np.dot(np.dot(Y.T,np.diag(lam)),X),full_matrices=False)

    lam_factor = np.sqrt(np.sum(lam_new**2)) 
    lam_new /= lam_factor ## normalization

    Tn_new = np.tensordot(np.tensordot(np.dot(VT,linalg.pinv(X)),Tn,(1,0)),np.dot(linalg.pinv(Y).T,U),(2,0))

    ## normalization to be dominant eigen value = 1
    Tn_new /= np.sqrt(np.abs(np.real(eig_val_R)))/lam_factor
                
    return Tn_new,lam_new,eig_val_R,eig_val_L

def Make_Canonical_form_2site_old(Tn1,Tn2,lam1,lam2):
    ## We assume uniforma iMPS with 2-site unit cell
    ## Calculate left and right dominant eigen vectors

    chi = Tn1.shape[0]

    Tn1_L = np.tensordot(np.diag(lam2),Tn1,(1,0))
    Tn2_L = np.tensordot(np.diag(lam1),Tn2,(1,0))
    Tn1_R = np.tensordot(Tn1,np.diag(lam1),(2,0))
    Tn2_R = np.tensordot(Tn2,np.diag(lam2),(2,0))
    def vL(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn1_L,(0,0)),Tn1_L.conj(),([0,1],[0,1]))
        v_new = np.tensordot(np.tensordot(v_new,Tn2_L,(0,0)),Tn2_L.conj(),([0,1],[0,1]))

        return v_new.reshape(chi**2)
    def Rv(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn2_R,(0,2)),Tn2_R.conj(),([0,2],[2,1]))
        v_new = np.tensordot(np.tensordot(v_new,Tn1_R,(0,2)),Tn1_R.conj(),([0,2],[2,1]))
        return v_new.reshape(chi**2)


    if chi > 1:    
        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=Rv)
        eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=vL)

        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_R = np.ones(1)
        eig_vec_L = np.ones(1)
        eig_val_R = Rv(eig_vec_R)
        eig_val_L = vL(eig_vec_L)

    print("eig_vals = "+repr(eig_val_L)+" " +repr(eig_val_R))
    UR, sR, VTR = linalg.svd(np.real(eig_vec_R).reshape(chi,chi),full_matrices=False)
    UL, sL, VTL = linalg.svd(np.real(eig_vec_L).reshape(chi,chi),full_matrices=False)

    X = np.dot(UR,np.diag(np.sqrt(sR)))
    Y = np.dot(UL,np.diag(np.sqrt(sL)))

    #U, lam_new, VT = linalg.svd(np.dot(np.dot(Y.T,np.diag(lam2)),X),full_matrices=False)
    U, lam_new, VT = linalg.svd(np.dot(Y.T,X),full_matrices=False)

    # U = U[:,:chi]
    # lam_new = lam_new[:chi]
    # VT = VT[:chi,:]

    lam2_factor = np.sqrt(np.sum(lam_new**2)) 
    lam2_new = lam_new / lam2_factor ## normalization

    Theta = np.tensordot(np.tensordot(np.diag(lam2),Tn1_R,(1,0)),Tn2_R,(2,0))

    Sigma = np.tensordot(np.tensordot(np.dot(np.dot(np.diag(lam2_new),VT),linalg.pinv(X)),Theta,(1,0)),np.dot(np.dot(linalg.pinv(Y).T,U),np.diag(lam2_new)),(3,0)).reshape(chi*Tn1.shape[1],-1)
    P,lam_new,Q = linalg.svd(Sigma,full_matrices=False)

    chi1 = len(lam1)
    
    P = P[:,:chi1].reshape(chi,Tn1.shape[1],chi1)
    lam_new = lam_new[:chi1]
    Q = Q[:chi1,:].reshape(chi1,Tn2.shape[1],chi)

    lam1_factor = np.sqrt(np.sum(lam_new**2)) 
    lam1_new = lam_new / lam1_factor ## normalization
    
    Tn1_new = np.tensordot(np.diag(1.0/lam2_new),P,(1,0))
    Tn2_new = np.tensordot(Q,np.diag(1.0/lam2_new),(2,0))
    
    
    ## normalization to be dominant eigen value = 1
    Tn1_new /= np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))
    Tn2_new /= np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))
                
    return Tn1_new,Tn2_new,lam1_new,lam2_new,eig_val_R,eig_val_L

def Make_Canonical_form_2site(Tn1,Tn2,lam1,lam2):
    ## We assume uniforma iMPS with 2-site unit cell
    ## Calculate left and right dominant eigen vectors

    chi = Tn1.shape[0]

    Tn1_L = np.tensordot(np.diag(lam2),Tn1,(1,0))
    Tn2_L = np.tensordot(np.diag(lam1),Tn2,(1,0))
    Tn1_R = np.tensordot(Tn1,np.diag(lam1),(2,0))
    Tn2_R = np.tensordot(Tn2,np.diag(lam2),(2,0))
    def vL(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn1_L,(0,0)),Tn1_L.conj(),([0,1],[0,1]))
        v_new = np.tensordot(np.tensordot(v_new,Tn2_L,(0,0)),Tn2_L.conj(),([0,1],[0,1]))

        return v_new.reshape(chi**2)
    def Rv(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn2_R,(0,2)),Tn2_R.conj(),([0,2],[2,1]))
        v_new = np.tensordot(np.tensordot(v_new,Tn1_R,(0,2)),Tn1_R.conj(),([0,2],[2,1]))
        return v_new.reshape(chi**2)


    if chi > 1:    
        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=Rv)
        eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=vL)

        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_R = np.ones(1)
        eig_vec_L = np.ones(1)
        eig_val_R = Rv(eig_vec_R)
        eig_val_L = vL(eig_vec_L)

    print("eig_vals = "+repr(eig_val_L)+" " +repr(eig_val_R))
    UR, sR, VTR = linalg.svd(np.real(eig_vec_R).reshape(chi,chi),full_matrices=False)
    UL, sL, VTL = linalg.svd(np.real(eig_vec_L).reshape(chi,chi),full_matrices=False)

    X = np.dot(UR,np.diag(np.sqrt(sR)))
    Y = np.dot(UL,np.diag(np.sqrt(sL)))

    U, lam_new, VT = linalg.svd(np.dot(np.dot(Y.T,np.diag(lam2)),X),full_matrices=False)
    #U, lam_new, VT = linalg.svd(np.dot(Y.T,X),full_matrices=False)

    # U = U[:,:chi]
    # lam_new = lam_new[:chi]
    # VT = VT[:chi,:]

    lam2_factor = np.sqrt(np.sum(lam_new**2)) 
    lam2_new = lam_new / lam2_factor ## normalization

    Theta = np.tensordot(Tn1_R,Tn2,(2,0))

    Sigma = np.tensordot(np.tensordot(np.dot(np.dot(np.diag(lam2_new),VT),linalg.pinv(X)),Theta,(1,0)),np.dot(np.dot(linalg.pinv(Y).T,U),np.diag(lam2_new)),(3,0)).reshape(chi*Tn1.shape[1],-1)
    P,lam_new,Q = linalg.svd(Sigma,full_matrices=False)

    chi1 = len(lam1)
    
    P = P[:,:chi1].reshape(chi,Tn1.shape[1],chi1)
    lam_new = lam_new[:chi1]
    Q = Q[:chi1,:].reshape(chi1,Tn2.shape[1],chi)

    lam1_factor = np.sqrt(np.sum(lam_new**2)) 
    lam1_new = lam_new / lam1_factor ## normalization
    
    Tn1_new = np.tensordot(np.diag(1.0/lam2_new),P,(1,0))
    Tn2_new = np.tensordot(Q,np.diag(1.0/lam2_new),(2,0))
    
    
    ## normalization to be dominant eigen value = 1
    Tn1_new /= np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))
    Tn2_new /= np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))
                
    return Tn1_new,Tn2_new,lam1_new,lam2_new,eig_val_R,eig_val_L


def Make_Canonical_form_symmetric(Tn,lam):
    ## We assume uniforma iMPS and left-right symmetry     
    ## Calculate left dominant eigen vector only

    chi = Tn.shape[0]

    Tn_L = np.tensordot(np.diag(lam),Tn,(1,0))
    def vL(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi,chi),Tn_L,(0,0)),Tn_L.conj(),([0,1],[0,1]))

        return v_new.reshape(chi**2)

    if chi > 1:    
        T_mat = spr_linalg.LinearOperator((chi**2,chi**2),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_L = np.ones(1)
        eig_val_L = vL(eig_vec_L)
        
    UL, sL, VTL = linalg.svd(np.real(eig_vec_L).reshape(chi,chi),full_matrices=False)

    Y = np.dot(UL,np.diag(np.sqrt(sL)))

    U, lam_new, VT = linalg.svd(np.dot(np.dot(Y.T,np.diag(lam)),Y.conj()),full_matrices=False)

    lam_factor = np.sqrt(np.sum(lam_new**2)) 
    lam_new /= lam_factor ## normalization

    Y_inv = linalg.pinv(Y)
    X_inv = Y_inv.conj()
    Tn_new = np.tensordot(np.tensordot(np.dot(VT,X_inv),Tn,(1,0)),np.dot(Y_inv.T,U),(2,0))

    ## normalization to be dominant eigen value = 1
    Tn_new /= np.sqrt(np.real(eig_val_L))/lam_factor
                
    return Tn_new,lam_new,eig_val_L


def Normalize_two_states(Tn_U,Tn_D,lam_U,lam_D):
    ## Not yet tested!!
    ## We assume uniforma iMPS
    ## Calculate dominant eigen value, and return Tn_D / eig

    chi_U = Tn_U.shape[0]
    chi_D = Tn_D.shape[0]
    
    lam_U_sq = np.sqrt(lam_U)
    lam_D_sq = np.sqrt(lam_D)

    Tn_U_mod = np.tensordot(np.tensordot(np.diag(lam_U_sq),Tn_U,(1,0)),np.diag(lam_U_sq),(2,0))
    Tn_D_mod = np.tensordot(np.tensordot(np.diag(lam_D_sq),Tn_D,(1,0)),np.diag(lam_D_sq),(2,0))
    
    def vL(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi_U,chi_D),Tn_U_mod,(0,0)),Tn_D_mod,([0,1],[0,1]))

        return v_new.reshape(chi_U*chi_D)

    if chi_U*chi_D > 1:    
        T_mat = spr_linalg.LinearOperator((chi_U*chi_D,chi_U*chi_D),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_L = np.ones(1)
        eig_val_L = vL(eig_vec_L)
        
    return Tn_D / np.real(eig_val_L), np.real(eig_val_L)

def Normalize_two_states_2site(Tn1_U,Tn2_U,Tn1_D,Tn2_D,lam1_U,lam2_U,lam1_D,lam2_D):
    ## Not yet tested!!
    ## We assume 2-site iMPS
    ## Calculate dominant eigen value, and return Tn1_D / eig

    chi_U = Tn1_U.shape[0]
    chi_D = Tn1_D.shape[0]
    
    lam1_U_sq = np.sqrt(lam1_U)
    lam1_D_sq = np.sqrt(lam1_D)
    lam2_U_sq = np.sqrt(lam2_U)
    lam2_D_sq = np.sqrt(lam2_D)

    Tn1_U_mod = np.tensordot(np.tensordot(np.diag(lam2_U_sq),Tn1_U,(1,0)),np.diag(lam1_U),(2,0))
    Tn2_U_mod = np.tensordot(Tn2_U,np.diag(lam2_U_sq),(2,0))
    Tn1_D_mod = np.tensordot(np.tensordot(np.diag(lam2_D_sq),Tn1_D,(1,0)),np.diag(lam1_D),(2,0))
    Tn2_D_mod = np.tensordot(Tn2_D,np.diag(lam2_D_sq),(2,0))

    def vL(v):
        v_new = np.tensordot(np.tensordot(v.reshape(chi_U,chi_D),Tn1_U_mod,(0,0)),Tn1_D_mod,([0,1],[0,1]))
        v_new = np.tensordot(np.tensordot(v_new,Tn2_U_mod,(0,0)),Tn2_D_mod,([0,1],[0,1]))

        return v_new.reshape(chi_U*chi_D)

    if chi_U*chi_D > 1:    
        T_mat = spr_linalg.LinearOperator((chi_U*chi_D,chi_U*chi_D),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_L = np.ones(1)
        eig_val_L = vL(eig_vec_L)

    return Tn1_D / np.real(eig_val_L), np.real(eig_val_L)


def iTEBD_MPO1_update(Tn,lam,MPO,chi_max,lam_epsilon=np.finfo(np.float64).eps,symmetry=False):
    ## We assume 1-site iMPS
    chi = Tn.shape[0]
    d = MPO.shape[0]

    lam_temp = np.kron(lam,np.ones(d))
    Tn_temp = np.tensordot(Tn,MPO,(1,1)).transpose(0,2,4,1,3).reshape(chi*d,MPO.shape[3],chi*d)

    ## transform to the canonical form 

    if symmetry:
        Tn_temp,lam_temp,eig_val_L = Make_Canonical_form_symmetric(Tn_temp,lam_temp)
    else:
        Tn_temp,lam_temp,eig_val_R,eig_val_L = Make_Canonical_form(Tn_temp,lam_temp)

    ## Truncation 
    chi_c = np.min([np.sum(lam_temp > lam_epsilon),chi_max])
    
    lam_new = lam_temp[:chi_c]/np.sqrt(np.sum(lam_temp[:chi_c]**2))

    truncation_error = np.sum(lam_temp[chi_c:]**2)

    Tn_new = Tn_temp[:chi_c,:,:chi_c]

    return Tn_new,lam_new,truncation_error


def iTEBD_MPO2_update_old(Tn1,Tn2,lam1,lam2,MPO1,MPO2,chi_max,lam_epsilon=np.finfo(np.float64).eps):
    ## We assume 2-site iMPS
    chi_l = Tn1.shape[0]
    chi_r = Tn2.shape[2]
    d_l = MPO1.shape[0]
    d_r = MPO2.shape[2]


    Tn1_lam_R = np.tensordot(Tn1,np.diag(lam1),(2,0))
    Tn2_lam_R = np.tensordot(Tn2,np.diag(lam2),(2,0))

    Tn1_lam_L = np.tensordot(np.diag(lam2),Tn1,(1,0))
    Tn2_lam_L = np.tensordot(np.diag(lam1),Tn2,(1,0))
    
    def vAB(v):
        v_temp = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_l,d_l,chi_l,d_l),Tn1_lam_L,(0,0)),MPO1,([0,3],[0,1])),MPO1.conj(),([1,4],[0,3])),Tn1_lam_L.conj(),([0,3],[0,1]))
        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v_temp,Tn2_lam_L,(0,0)),MPO2,([0,3],[0,1])),MPO2.conj(),([0,4],[0,3])),Tn2_lam_L.conj(),([0,3],[0,1]))

        return v_new.transpose(0,1,3,2).reshape(-1)
    def ABv(v):
        v_temp = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_r,d_r,chi_r,d_r),Tn2_lam_R,(0,2)),MPO2,([0,4],[2,1])),MPO2.conj(),([1,4],[2,3])),Tn2_lam_R.conj(),([0,4],[2,1]))

        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v_temp,Tn1_lam_R,(0,2)),MPO1,([0,4],[2,1])),MPO1.conj(),([0,4],[2,3])),Tn1_lam_R.conj(),([0,4],[2,1]))
                                                               
        return v_new.transpose(0,1,3,2).reshape(-1)


    T_mat = spr_linalg.LinearOperator(((chi_r*d_r)**2,(chi_r*d_r)**2),matvec=ABv)
    eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

    T_mat = spr_linalg.LinearOperator(((chi_l*d_l)**2,(chi_l*d_l)**2),matvec=vAB)

    eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)

    UR, sR, VTR = linalg.svd(np.real(eig_vec_R).reshape(chi_r*d_r,chi_r*d_r),full_matrices=False)
    UL, sL, VTL = linalg.svd(np.real(eig_vec_L).reshape(chi_l*d_l,chi_l*d_l),full_matrices=False)

    X = np.dot(UR,np.diag(np.sqrt(sR)))
    Y = np.dot(UL,np.diag(np.sqrt(sL)))
    
    U, s, VT = linalg.svd(np.dot(Y.T,X),full_matrices=False)

    ## Truncation of new lam2
    lam2_factor = np.sqrt(np.sum(s**2))
    s /=  lam2_factor## tentative normalization for calculation of the truncation error
    
    chi_c2 = np.min([np.sum(s > lam_epsilon),chi_max])

    lam2_new = s [:chi_c2]/np.sqrt(np.sum(s[:chi_c2]**2))
    U = U[:,:chi_c2]
    VT = VT[:chi_c2,:]

    truncation_error2 = np.sum(s[chi_c2:]**2)

    ## Calculation of new lam1
    
    Theta = np.tensordot(np.tensordot(np.tensordot(np.tensordot(np.diag(lam2),Tn1_lam_R,(1,0)),Tn2_lam_R,(2,0)),MPO1,(1,1)),MPO2,([1,4],[1,0])).transpose(0,2,3,5,1,4).reshape(chi_l*d_l,MPO1.shape[3],MPO2.shape[3],chi_r*d_r)
    
    Sigma = np.tensordot(np.tensordot(np.dot(np.dot(np.diag(lam2_new),VT),linalg.pinv(X)), Theta, (1,0)), np.dot(np.dot(np.diag(lam2_new),U.T),linalg.pinv(Y)),(3,1))

    P, s, Q = linalg.svd(Sigma.reshape(chi_c2*MPO1.shape[3],MPO2.shape[3]*chi_c2),full_matrices=False)

    lam1_factor = np.sqrt(np.sum(s**2))
    s /=  lam1_factor## tentative normalization for calculation of the truncation error
    
    ## Truncation of new lam1
    chi_c1 = np.min([np.sum(s > lam_epsilon),chi_max])

    lam1_new = s[:chi_c1]/np.sqrt(np.sum(s[:chi_c1]**2))

    truncation_error1 = np.sum(s[chi_c1:]**2)
    
    Tn1_new = np.tensordot(np.diag(1.0/lam2_new),P[:,:chi_c1].reshape(chi_c2,-1,chi_c1),(1,0)) /np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))
    Tn2_new = np.tensordot(Q[:chi_c1,:].reshape(chi_c1,-1,chi_c2),np.diag(1.0/lam2_new),(2,0)) /np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))

    return Tn1_new,Tn2_new,lam1_new,lam2_new,truncation_error1,truncation_error2

def iTEBD_MPO2_update(Tn1,Tn2,lam1,lam2,MPO1,MPO2,chi_max,lam_epsilon=np.finfo(np.float64).eps):
    ## We assume 2-site iMPS
    chi_l = Tn1.shape[0]
    chi_r = Tn2.shape[2]
    d_l = MPO1.shape[0]
    d_r = MPO2.shape[2]


    Tn1_lam_R = np.tensordot(Tn1,np.diag(lam1),(2,0))
    Tn2_lam_R = np.tensordot(Tn2,np.diag(lam2),(2,0))

    Tn1_lam_L = np.tensordot(np.diag(lam2),Tn1,(1,0))
    Tn2_lam_L = np.tensordot(np.diag(lam1),Tn2,(1,0))
    
    def vAB(v):
        v_temp = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_l,d_l,chi_l,d_l),Tn1_lam_L,(0,0)),MPO1,([0,3],[0,1])),MPO1.conj(),([1,4],[0,3])),Tn1_lam_L.conj(),([0,3],[0,1]))
        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v_temp,Tn2_lam_L,(0,0)),MPO2,([0,3],[0,1])),MPO2.conj(),([0,4],[0,3])),Tn2_lam_L.conj(),([0,3],[0,1]))

        return v_new.transpose(0,1,3,2).reshape(-1)
    def ABv(v):
        v_temp = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_r,d_r,chi_r,d_r),Tn2_lam_R,(0,2)),MPO2,([0,4],[2,1])),MPO2.conj(),([1,4],[2,3])),Tn2_lam_R.conj(),([0,4],[2,1]))

        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v_temp,Tn1_lam_R,(0,2)),MPO1,([0,4],[2,1])),MPO1.conj(),([0,4],[2,3])),Tn1_lam_R.conj(),([0,4],[2,1]))
                                                               
        return v_new.transpose(0,1,3,2).reshape(-1)


    T_mat = spr_linalg.LinearOperator(((chi_r*d_r)**2,(chi_r*d_r)**2),matvec=ABv)
    eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

    T_mat = spr_linalg.LinearOperator(((chi_l*d_l)**2,(chi_l*d_l)**2),matvec=vAB)

    eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)

    UR, sR, VTR = linalg.svd(np.real(eig_vec_R).reshape(chi_r*d_r,chi_r*d_r),full_matrices=False)
    UL, sL, VTL = linalg.svd(np.real(eig_vec_L).reshape(chi_l*d_l,chi_l*d_l),full_matrices=False)

    X = np.dot(UR,np.diag(np.sqrt(sR)))
    Y = np.dot(UL,np.diag(np.sqrt(sL)))
    
    U, s, VT = linalg.svd(np.dot(np.dot(Y.T,np.diag(np.kron(lam2,np.ones(d_l)))),X),full_matrices=False)

    ## Truncation of new lam2
    lam2_factor = np.sqrt(np.sum(s**2))
    s /=  lam2_factor## tentative normalization for calculation of the truncation error
    
    chi_c2 = np.min([np.sum(s > lam_epsilon),chi_max])

    lam2_new = s [:chi_c2]/np.sqrt(np.sum(s[:chi_c2]**2))
    U = U[:,:chi_c2]
    VT = VT[:chi_c2,:]

    truncation_error2 = np.sum(s[chi_c2:]**2)

    ## Calculation of new lam1
    
    Theta = np.tensordot(np.tensordot(np.tensordot(Tn1_lam_R,Tn2,(2,0)),MPO1,(1,1)),MPO2,([1,4],[1,0])).transpose(0,2,3,5,1,4).reshape(chi_l*d_l,MPO1.shape[3],MPO2.shape[3],chi_r*d_r)
    
    Sigma = np.tensordot(np.tensordot(np.dot(np.dot(np.diag(lam2_new),VT),linalg.pinv(X)), Theta, (1,0)), np.dot(np.dot(np.diag(lam2_new),U.T),linalg.pinv(Y)),(3,1))

    P, s, Q = linalg.svd(Sigma.reshape(chi_c2*MPO1.shape[3],MPO2.shape[3]*chi_c2),full_matrices=False)

    lam1_factor = np.sqrt(np.sum(s**2))
    s /=  lam1_factor## tentative normalization for calculation of the truncation error
    
    ## Truncation of new lam1
    chi_c1 = np.min([np.sum(s > lam_epsilon),chi_max])

    lam1_new = s[:chi_c1]/np.sqrt(np.sum(s[:chi_c1]**2))

    truncation_error1 = np.sum(s[chi_c1:]**2)
    
    Tn1_new = np.tensordot(np.diag(1.0/lam2_new),P[:,:chi_c1].reshape(chi_c2,-1,chi_c1),(1,0)) /np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))
    Tn2_new = np.tensordot(Q[:chi_c1,:].reshape(chi_c1,-1,chi_c2),np.diag(1.0/lam2_new),(2,0)) /np.sqrt(np.sqrt(np.abs(np.real(eig_val_R)))/(lam1_factor*lam2_factor))

    return Tn1_new,Tn2_new,lam1_new,lam2_new,truncation_error1,truncation_error2

def iTEBD_MPO2_update_canonical(Tn1,Tn2,lam1,lam2,MPO1,MPO2,chi_max,lam_epsilon=np.finfo(np.float64).eps):
    ## We assume 2-site iMPS
    chi1 = Tn1.shape[2]
    chi2 = Tn2.shape[2]
    d1 = MPO1.shape[2]
    d2 = MPO2.shape[2]

    lam1_temp = np.kron(lam1,np.ones(d1))
    lam2_temp = np.kron(lam2,np.ones(d1))
    Tn1_temp = np.tensordot(Tn1,MPO1,(1,1)).transpose(0,2,4,1,3).reshape(chi2*d2,MPO1.shape[3],chi1*d1)
    Tn2_temp = np.tensordot(Tn2,MPO2,(1,1)).transpose(0,2,4,1,3).reshape(chi1*d1,MPO2.shape[3],chi2*d2)

    ## transform to the canonical form 

    Tn1_temp, Tn2_temp,lam1_temp,lam2_temp,eig_val_R,eig_val_L = Make_Canonical_form_2site(Tn1_temp,Tn2_temp,lam1_temp,lam2_temp)
    #Tn1_temp, Tn2_temp,lam1_temp,lam2_temp,eig_val_R,eig_val_L = Make_Canonical_form_2site_new(Tn1_temp,Tn2_temp,lam1_temp,lam2_temp)
    
    ## Truncation 
    chi_c1 = np.min([np.sum(lam1_temp > lam_epsilon),chi_max])
    chi_c2 = np.min([np.sum(lam2_temp > lam_epsilon),chi_max])
    
    lam1_new = lam1_temp[:chi_c1]/np.sqrt(np.sum(lam1_temp[:chi_c1]**2))
    lam2_new = lam2_temp[:chi_c2]/np.sqrt(np.sum(lam2_temp[:chi_c2]**2))

    truncation_error1 = np.sum(lam1_temp[chi_c1:]**2)
    truncation_error2 = np.sum(lam2_temp[chi_c2:]**2)

    Tn1_new = Tn1_temp[:chi_c2,:,:chi_c1]
    Tn2_new = Tn2_temp[:chi_c1,:,:chi_c2]

    return Tn1_new,Tn2_new,lam1_new,lam2_new,truncation_error1,truncation_error2

    
def Calc_dominant_vectors(Tn_U,Tn_D,lam_U,lam_D,local_T):
    ## We assume inner product of U and D states is 1.
    chi_U = Tn_U.shape[0]
    chi_D = Tn_D.shape[0]
    d = local_T.shape[2]
    
    lam_U_sq = np.sqrt(lam_U)
    lam_D_sq = np.sqrt(lam_D)

    Tn_U_mod = np.tensordot(np.tensordot(np.diag(lam_U_sq),Tn_U,(1,0)),np.diag(lam_U_sq),(2,0))
    Tn_D_mod = np.tensordot(np.tensordot(np.diag(lam_D_sq),Tn_D,(1,0)),np.diag(lam_D_sq),(2,0))

    
    def vL(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d,chi_D),Tn_U_mod,(0,0)),local_T,([0,2],[0,1])),Tn_D_mod,([0,3],[0,1]))

        return v_new.reshape(chi_U*d*chi_D)
    def Rv(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d,chi_D),Tn_U_mod,(0,2)),local_T,([0,3],[2,1])),Tn_D_mod,([0,3],[2,1]))
        return v_new.reshape(chi_U*d*chi_D)
    
    if chi_U*d*chi_D > 1:    
        T_mat = spr_linalg.LinearOperator((chi_U*d*chi_D,chi_U*d*chi_D),matvec=Rv)
        eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

        T_mat = spr_linalg.LinearOperator((chi_U*d*chi_D,chi_U*d*chi_D),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_R = np.ones(1)
        eig_vec_L = np.ones(1)
        eig_val_R = Rv(eig_vec_R)
        eig_val_L = vL(eig_vec_L)

    return eig_val_R,eig_val_L, eig_vec_R.reshape(chi_U,d,chi_D), eig_vec_L.reshape(chi_U,d,chi_D)

def Calc_dominant_vectors_2site(Tn1_U,Tn2_U,Tn1_D,Tn2_D,lam1_U,lam2_U,lam1_D,lam2_D,local_T1,local_T2):
    ## We assume inner product of U and D states is 1. 2-site unit cell
    chi_U = Tn1_U.shape[0]
    chi_D = Tn1_D.shape[0]
    d = local_T2.shape[2]
    
    lam1_U_sq = np.sqrt(lam1_U)
    lam1_D_sq = np.sqrt(lam1_D)
    lam2_U_sq = np.sqrt(lam2_U)
    lam2_D_sq = np.sqrt(lam2_D)

    Tn1_U_mod = np.tensordot(np.tensordot(np.diag(lam2_U_sq),Tn1_U,(1,0)),np.diag(lam1_U),(2,0))
    Tn2_U_mod = np.tensordot(Tn2_U,np.diag(lam2_U_sq),(2,0))
    Tn1_D_mod = np.tensordot(np.tensordot(np.diag(lam2_D_sq),Tn1_D,(1,0)),np.diag(lam1_D),(2,0))
    Tn2_D_mod = np.tensordot(Tn2_D,np.diag(lam2_D_sq),(2,0))
    
    def vL(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d,chi_D),Tn1_U_mod,(0,0)),local_T1,([0,2],[0,1])),Tn1_D_mod,([0,3],[0,1]))
        v_new = np.tensordot(np.tensordot(np.tensordot(v_new,Tn2_U_mod,(0,0)),local_T2,([0,2],[0,1])),Tn2_D_mod,([0,3],[0,1]))

        return v_new.reshape(chi_U*d*chi_D)
    def Rv(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d,chi_D),Tn2_U_mod,(0,2)),local_T2,([0,3],[2,1])),Tn2_D_mod,([0,3],[2,1]))
        v_new = np.tensordot(np.tensordot(np.tensordot(v_new,Tn1_U_mod,(0,2)),local_T1,([0,3],[2,1])),Tn1_D_mod,([0,3],[2,1]))
        return v_new.reshape(chi_U*d*chi_D)
    
    if chi_U*d*chi_D > 1:    
        T_mat = spr_linalg.LinearOperator((chi_U*d*chi_D,chi_U*d*chi_D),matvec=Rv)
        eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

        T_mat = spr_linalg.LinearOperator((chi_U*d*chi_D,chi_U*d*chi_D),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_R = np.ones(1)
        eig_vec_L = np.ones(1)
        eig_val_R = Rv(eig_vec_R)
        eig_val_L = vL(eig_vec_L)

    return eig_val_R,eig_val_L, eig_vec_R.reshape(chi_U,d,chi_D), eig_vec_L.reshape(chi_U,d,chi_D)

def Calc_dominant_vectors_2site_2layer(Tn1_U,Tn2_U,Tn1_D,Tn2_D,lam1_U,lam2_U,lam1_D,lam2_D,local_T1,local_T2,local_T3,local_T4):
    ## We assume inner product of U and D states is 1. 2-site unit cell
    chi_U = Tn1_U.shape[0]
    chi_D = Tn1_D.shape[0]
    d2 = local_T2.shape[2]
    d4 = local_T4.shape[2]
    
    lam1_U_sq = np.sqrt(lam1_U)
    lam1_D_sq = np.sqrt(lam1_D)
    lam2_U_sq = np.sqrt(lam2_U)
    lam2_D_sq = np.sqrt(lam2_D)

    Tn1_U_mod = np.tensordot(np.tensordot(np.diag(lam2_U_sq),Tn1_U,(1,0)),np.diag(lam1_U),(2,0))
    Tn2_U_mod = np.tensordot(Tn2_U,np.diag(lam2_U_sq),(2,0))
    Tn1_D_mod = np.tensordot(np.tensordot(np.diag(lam2_D_sq),Tn1_D,(1,0)),np.diag(lam1_D),(2,0))
    Tn2_D_mod = np.tensordot(Tn2_D,np.diag(lam2_D_sq),(2,0))
    
    def vL(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d2,d4,chi_D),Tn1_U_mod,(0,0)),local_T1,([0,3],[0,1])),local_T3,([0,4],[0,1])),Tn1_D_mod,([0,4],[0,1]))
        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v_new,Tn2_U_mod,(0,0)),local_T2,([0,3],[0,1])),local_T4,([0,4],[0,1])),Tn2_D_mod,([0,4],[0,1]))

        return v_new.reshape(chi_U*d2*d4*chi_D)
    def Rv(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d2,d4,chi_D),Tn2_U_mod,(0,2)),local_T2,([0,4],[2,1])),local_T4,([0,4],[2,1])),Tn2_D_mod,([0,4],[2,1]))
        v_new = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v_new,Tn1_U_mod,(0,2)),local_T1,([0,4],[2,1])),local_T3,([0,4],[2,1])),Tn1_D_mod,([0,4],[2,1]))
        return v_new.reshape(chi_U*d2*d4*chi_D)
    
    if chi_U*d2*d4*chi_D > 1:    
        T_mat = spr_linalg.LinearOperator((chi_U*d2*d4*chi_D,chi_U*d2*d4*chi_D),matvec=Rv)
        eig_val_R,eig_vec_R = spr_linalg.eigs(T_mat,k=1)

        T_mat = spr_linalg.LinearOperator((chi_U*d2*d4*chi_D,chi_U*d2*d4*chi_D),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigs(T_mat,k=1)
    else:
        eig_vec_R = np.ones(1)
        eig_vec_L = np.ones(1)
        eig_val_R = Rv(eig_vec_R)
        eig_val_L = vL(eig_vec_L)

    return eig_val_R,eig_val_L, eig_vec_R.reshape(chi_U,d2,d4,chi_D), eig_vec_L.reshape(chi_U,d2,d4,chi_D)

def Calc_dominant_vectors_symmetric(Tn_U,Tn_D,lam_U,lam_D,local_T):
    ## We assume inner product of U and D states is 1.
    ## We assume left-right symmetry
    chi_U = Tn_U.shape[0]
    chi_D = Tn_D.shape[0]
    d = local_T.shape[2]
    
    lam_U_sq = np.sqrt(lam_U)
    lam_D_sq = np.sqrt(lam_D)

    Tn_U_mod = np.tensordot(np.tensordot(np.diag(lam_U_sq),Tn_U,(1,0)),np.diag(lam_U_sq),(2,0))
    Tn_D_mod = np.tensordot(np.tensordot(np.diag(lam_D_sq),Tn_D,(1,0)),np.diag(lam_D_sq),(2,0))

    
    def vL(v):
        v_new = np.tensordot(np.tensordot(np.tensordot(v.reshape(chi_U,d,chi_D),Tn_U_mod,(0,0)),local_T,([0,2],[0,1])),Tn_D_mod,([0,3],[0,1]))

        return v_new.reshape(chi_U*d*chi_D)
    
    if chi_U*d*chi_D > 1:    
        T_mat = spr_linalg.LinearOperator((chi_U*d*chi_D,chi_U*d*chi_D),matvec=vL)
        eig_val_L,eig_vec_L = spr_linalg.eigsh(T_mat,k=1)
    else:
        eig_vec_L = np.ones(1)
        eig_val_L = vL(eig_vec_L)

    return eig_val_L,  eig_vec_L.reshape(chi_U,d,chi_D)

def Calc_1site(Tn_U,Tn_D,lam_U,lam_D,local_T,vec_R,vec_L):
    chi_U = Tn_U.shape[0]
    chi_D = Tn_D.shape[0]
    d = local_T.shape[2]
    
    lam_U_sq = np.sqrt(lam_U)
    lam_D_sq = np.sqrt(lam_D)
    
    Tn_U_mod = np.tensordot(np.tensordot(np.diag(lam_U_sq),Tn_U,(1,0)),np.diag(lam_U_sq),(2,0))
    Tn_D_mod = np.tensordot(np.tensordot(np.diag(lam_D_sq),Tn_D,(1,0)),np.diag(lam_D_sq),(2,0))

    return np.tensordot(np.tensordot(np.tensordot(np.tensordot(vec_L,Tn_U_mod,(0,0)),local_T,([0,2],[0,1])),Tn_D_mod,([0,3],[0,1])),vec_R,([0,1,2],[0,1,2]))
                                                                                                        
def Calc_2site(Tn1_U,Tn2_U,Tn1_D,Tn2_D,lam1_U,lam2_U,lam1_D,lam2_D,local_T1,local_T2,vec_R,vec_L):
    chi_U = Tn1_U.shape[0]
    chi_D = Tn1_D.shape[0]
    d = local_T2.shape[2]
    
    lam1_U_sq = np.sqrt(lam1_U)
    lam1_D_sq = np.sqrt(lam1_D)
    lam2_U_sq = np.sqrt(lam2_U)
    lam2_D_sq = np.sqrt(lam2_D)

    Tn1_U_mod = np.tensordot(np.tensordot(np.diag(lam2_U_sq),Tn1_U,(1,0)),np.diag(lam1_U),(2,0))
    Tn2_U_mod = np.tensordot(Tn2_U,np.diag(lam2_U_sq),(2,0))
    Tn1_D_mod = np.tensordot(np.tensordot(np.diag(lam2_D_sq),Tn1_D,(1,0)),np.diag(lam1_D),(2,0))
    Tn2_D_mod = np.tensordot(Tn2_D,np.diag(lam2_D_sq),(2,0))

    v = np.tensordot(np.tensordot(np.tensordot(vec_L,Tn1_U_mod,(0,0)),local_T1,([0,2],[0,1])),Tn1_D_mod,([0,3],[0,1]))
    v = np.tensordot(np.tensordot(np.tensordot(v,Tn2_U_mod,(0,0)),local_T2,([0,2],[0,1])),Tn2_D_mod,([0,3],[0,1]))

    
    return np.tensordot(v,vec_R,([0,1,2],[0,1,2]))

def Calc_2site_2layer(Tn1_U,Tn2_U,Tn1_D,Tn2_D,lam1_U,lam2_U,lam1_D,lam2_D,local_T1,local_T2,local_T3,local_T4,vec_R,vec_L):
    chi_U = Tn1_U.shape[0]
    chi_D = Tn1_D.shape[0]
    d2 = local_T2.shape[2]
    d4 = local_T4.shape[2]
    
    lam1_U_sq = np.sqrt(lam1_U)
    lam1_D_sq = np.sqrt(lam1_D)
    lam2_U_sq = np.sqrt(lam2_U)
    lam2_D_sq = np.sqrt(lam2_D)

    Tn1_U_mod = np.tensordot(np.tensordot(np.diag(lam2_U_sq),Tn1_U,(1,0)),np.diag(lam1_U),(2,0))
    Tn2_U_mod = np.tensordot(Tn2_U,np.diag(lam2_U_sq),(2,0))
    Tn1_D_mod = np.tensordot(np.tensordot(np.diag(lam2_D_sq),Tn1_D,(1,0)),np.diag(lam1_D),(2,0))
    Tn2_D_mod = np.tensordot(Tn2_D,np.diag(lam2_D_sq),(2,0))

    v = np.tensordot(np.tensordot(np.tensordot(np.tensordot(vec_L,Tn1_U_mod,(0,0)),local_T1,([0,3],[0,1])),local_T3,([0,4],[0,1])),Tn1_D_mod,([0,4],[0,1]))
    v = np.tensordot(np.tensordot(np.tensordot(np.tensordot(v,Tn2_U_mod,(0,0)),local_T2,([0,3],[0,1])),local_T4,([0,4],[0,1])),Tn2_D_mod,([0,4],[0,1]))

    
    return np.tensordot(v,vec_R,([0,1,2,3],[0,1,2,3]))
    
