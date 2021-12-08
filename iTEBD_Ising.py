import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
from iTEBD_classical import * 

#input from command line
def parse_args():
    Tc = 2.0/np.log(1.0+np.sqrt(2.0))
    parser = argparse.ArgumentParser(description='iTEBD simulation for Square lattice Ising model')
    parser.add_argument('--chi', metavar='chi',dest='chi', type=int, default=4,
                        help='set bond dimension chi for iTEBD. (default: chi=4)')
    parser.add_argument('--seed', metavar='seed',dest='seed', type=int, default=None,
                        help='set random seed. (default: seed = None)')
    parser.add_argument('--max_itr', metavar='max_itr',dest='max_itr', type=int, default=100,
                         help='set maximum iteration. (default: max_itr=100)')
    parser.add_argument('--epsilon', metavar='epsilon',dest='epsilon', type=float, default=1e-12,
                        help='Condition to stop the iteration. (default: epsilon=1e-12)')
    parser.add_argument('--lam_epsilon', metavar='lam_epsilon',dest='lam_epsilon', type=float, default=1e-12,
                        help='Epsilon for neglecting singular values. (default: lam_epsilon=1e-12)')
    parser.add_argument('-T', metavar='T',dest='T', type=float, default=Tc,
                        help='set Temperature. (default: T=Tc)')
    parser.add_argument('--append', action='store_const', const=True,
                        default=False, help='Output is added to the existing file.(default: append=False)')

    return parser.parse_args()


def Free_Energy_exact_2D_Ising(T):
    import scipy.integrate as integrate    
    def integrant(x,T):
        k = 1.0/np.sinh(2.0/T)**2
        k1 = 2.0*np.sqrt(k)/(1.0+k)
        result = np.log(2*(np.cosh(2.0/T)**2 + (k+1)/k*np.sqrt(1.0-k1**2*np.sin(x)**2)))
        return result

    k = 1.0/np.sinh(2.0/T)**2
    x,err =  integrate.quad(integrant, 0, np.pi*0.5, args=(T,),epsabs=1e-12,epsrel=1e-12)
    result = -T *x/np.pi

    return result,err * T/np.pi

def Mag_exact_2D_Ising(T):
    return (1-1.0/np.sinh(2/T)**4)**0.125

def initialize_A_site_tensor(T):
    # Make initial tensor of square lattice Ising model at a temperature T
    A =np.zeros((2,2,2,2),dtype=np.float64)

    ch = np.cosh(1.0/T)
    sh = np.sinh(1.0/T)
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                for l in range(0,2):

                    sum_index = i+j+k+l
                    if sum_index == 4:
                                            
                        A[i,j,k,l] = 2.0 * sh**2
                    elif sum_index == 2:
                        A[i,j,k,l] = 2.0 * ch * sh
                    elif sum_index == 0:
                        A[i,j,k,l] = 2.0 * ch**2
                                                
    #factor = Trace_tensor(A)
    factor = 1
    
    A /= factor
    
    return A,factor


def initialize_impurity_site_tensor(T, factor):
    # Make initial tensor of square lattice Ising model at a temperature T
    Ai =np.zeros((2,2,2,2),dtype=np.float64)

    ch = np.cosh(1.0/T)
    sh = np.sinh(1.0/T)

    csh = np.sqrt(ch*sh)
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                for l in range(0,2):                    
                    sum_index = i+j+k+l                    

                    if sum_index == 1:
                        Ai[i,j,k,l] = 2.0 * ch * csh

                    elif sum_index == 3:
                        Ai[i,j,k,l] = 2.0 * sh * csh
    ## normalization of tensor

    Ai /= factor
    
    return Ai
    
    
def Trace_tensor(A):
    # contraction of a single Tensor with the perioic boundary condition
    Trace_A = np.trace(A,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

def main():
    ## read params from command line
    args = parse_args()

    T = args.T
    chi_max = args.chi
    max_itr = args.max_itr
    epsilon = args.epsilon
    lam_epsilon = args.lam_epsilon
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)


    append_flag = args.append

    tag = "_chi"+str(chi_max)
    if append_flag:
        file_output = open("Ising_output"+tag+".dat","a")
    else:
        file_output = open("Ising_output"+tag+".dat","w")

    file_output.write("# T = "+repr(T)+"\n")
    file_output.write("# chi_max = "+repr(chi_max)+"\n")
    file_output.write("# max_itr = "+repr(max_itr)+"\n")
    file_output.write("# epsilon = "+repr(epsilon)+"\n")
    file_output.write("# lam_epsilon = "+repr(lam_epsilon)+"\n")
    file_output.write("# seed = "+repr(seed)+"\n")
    file_output.write("# append_flag = "+repr(append_flag)+"\n")
    file_output.write("## T, f=F/N, m=M/N,  f_exact, m_exact, iteration number, xi from left eigen value, xi from right eigen value"+"\n")



    
    ## Exact values
    f_ex,err = Free_Energy_exact_2D_Ising(T)
    m_ex = Mag_exact_2D_Ising(T)
        
    local_T,factor = initialize_A_site_tensor(T)
    local_T_mag = initialize_impurity_site_tensor(T,factor)

    
    Tn_init = (np.random.rand(2)-0.5).reshape(1,2,1)
    lam_init = np.ones(1)

    Tn, lam, eig_val_L = Make_Canonical_form_symmetric(Tn_init,lam_init)

    delta_lam = 1.0
    itr = 0
    while ( delta_lam > epsilon and itr < max_itr):        
        Tn_new,lam_new,truncation_error = iTEBD_MPO1_update(Tn,lam,local_T,chi_max,lam_epsilon,symmetry=True)

        #eig_R, eig_L, vec_R, vec_L = Calc_dominant_vectors(Tn_new,Tn_new,lam_new,lam_new,local_T
        eig_L, vec_L = Calc_dominant_vectors_symmetric(Tn_new,Tn_new,lam_new,lam_new,local_T)

        vec_R = vec_L.conj()
        norm = Calc_1site(Tn_new,Tn_new,lam_new,lam_new,local_T,vec_R,vec_L)
        mag = Calc_1site(Tn_new,Tn_new,lam_new,lam_new,local_T_mag,vec_R,vec_L)/norm
        
        ## calc singular value difference to check the convegence
        chi_comp = np.min([len(lam_new),len(lam)])
        delta_lam = np.sqrt(np.sum((lam_new[:chi_comp]-lam[:chi_comp])**2))

        print(repr(itr)+" "  +repr(-T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(mag))+" " +repr(f_ex)+" " +repr(m_ex)+" "+repr(delta_lam)+ " "+ repr(truncation_error))
        
        Tn = Tn_new
        lam = lam_new
              
        itr += 1

    print("## calculation has finised at "+repr(itr-1)+" steps")
    print(repr(T)+" "+ repr(-T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(mag))+" " +repr(f_ex)+" " +repr(m_ex)+" "+repr(itr))

    print("## calculate correlation length")
    ## calculate correlation length
    xi_L, xi_R = Calc_correlation_length(Tn,lam,1)
    print(repr(T)+" "+ repr(xi_L[0])+" " +repr(xi_R[0]))

    file_output.write(repr(T)+" "+ repr(-T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(mag))+" " +repr(f_ex)+" " +repr(m_ex)+" "+repr(itr)+ " " + repr(xi_L[0])+" " +repr(xi_R[0])+"\n")
                        
if __name__ == "__main__":
    main()

    
