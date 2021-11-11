import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
from iTEBD_classical import * 

#input from command line
def parse_args():
    Tc = 2.0/np.log(1.0+np.sqrt(2.0))
    parser = argparse.ArgumentParser(description='iTEBD simulation for Square lattice fully packed loop model')
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
    parser.add_argument('-n', metavar='n',dest='n', type=float, default=1.0,
                        help='set fugacity. (default: n=1.0)')
    parser.add_argument('--append', action='store_const', const=True,
                        default=False, help='Output is added to the existing file.(default: append=False)')

    return parser.parse_args()


def initialize_A_site_tensor(n):
    # Make initial tensor of square lattice Ising model at a temperature T
    A =np.zeros((2,2,2,2),dtype=np.float64)

    A[1,1,0,0] = n
    A[0,1,1,0] = n
    A[0,0,1,1] = n
    A[1,0,0,1] = n
    A[1,0,1,0] = n
    A[0,1,0,1] = n
                                                
    factor = Trace_tensor(A)
    
    A /= factor
    
    return A,factor


    
def Trace_tensor(A):
    # contraction of a single Tensor with the perioic boundary condition
    Trace_A = np.trace(A,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

def main():
    ## read params from command line
    args = parse_args()

    n = args.n
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
        file_output = open("Loop_output"+tag+".dat","a")
    else:
        file_output = open("Loop_output"+tag+".dat","w")

    file_output.write("# n = "+repr(n)+"\n")
    file_output.write("# chi_max = "+repr(chi_max)+"\n")
    file_output.write("# max_itr = "+repr(max_itr)+"\n")
    file_output.write("# epsilon = "+repr(epsilon)+"\n")
    file_output.write("# lam_epsilon = "+repr(lam_epsilon)+"\n")
    file_output.write("# seed = "+repr(seed)+"\n")
    file_output.write("# append_flag = "+repr(append_flag)+"\n")

    ## Exact values
    f_ex = -1.5 * np.log(4.0/3.0)
        
    local_T,factor = initialize_A_site_tensor(n)

    
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
        
        ## calc singular value difference to check the convegence
        chi_comp = np.min([len(lam_new),len(lam)])
        delta_lam = np.sqrt(np.sum((lam_new[:chi_comp]-lam[:chi_comp])**2))

        print(repr(itr)+" "  +repr(-np.log(np.real(eig_L[0] * factor)))+ " " +repr(f_ex)+" "+repr(delta_lam)+ " "+ repr(truncation_error))
        
        Tn = Tn_new
        lam = lam_new
              
        itr += 1

    print("## calculation has finised at "+repr(itr-1)+" steps")
    print(repr(n)+" "+ repr(-np.log(np.real(eig_L[0] * factor)))+" " +repr(f_ex)+" "+repr(itr))
    file_output.write(repr(n)+" "+ repr(-np.log(np.real(eig_L[0] * factor)))+" " +repr(f_ex)+" "+repr(itr)+"\n")
                        
if __name__ == "__main__":
    main()

    
