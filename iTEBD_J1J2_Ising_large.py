import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
from iTEBD_classical import * 

#input from command line
def parse_args():
    Tc = 2.0/np.log(1.0+np.sqrt(2.0))
    parser = argparse.ArgumentParser(description='iTEBD simulation for Square lattice J1-J2 Ising model')
    parser.add_argument('--chi', metavar='chi',dest='chi', type=int, default=4,
                        help='set bond dimension chi for iTEBD. (default: chi=4)')
    parser.add_argument('--J2', metavar='J2',dest='J2', type=float, default=0.0,
                        help='set J2. Note J1 is fixed at J1 = -1. (default: J2=0)')
    parser.add_argument('--seed', metavar='seed',dest='seed', type=int, default=None,
                        help='set random seed. (default: seed = None)')
    parser.add_argument('--initial_type', metavar='initial_type',dest='initial_type', type=int, default=1,
                         help='set initial_type (0:random, 1:mx). (default: initial_type=1)')
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


def initialize_A_site_tensor(T,J2):
    # Make initial tensor of square lattice Ising model at a temperature T
    A =np.zeros((2,2,2,2),dtype=np.float64)
    S =np.zeros((2,2,2,2),dtype=np.float64)

    for i in range(0,2):
        S[i,i,i,i] = 1.0
        si = (i - 0.5) * 2
        for j in range(0,2):
            sj = (j - 0.5) * 2
            for k in range(0,2):
                sk = (k - 0.5) * 2
                for l in range(0,2):
                    sl = (l - 0.5) * 2
                    A[i,j,k,l] = np.exp((0.5 * (si*sj + sj*sk + sk*sl + sl*si) -J2 * (si*sk + sj*sl))/T)


    AS = np.tensordot(A,S,(3,1))
    SA = np.tensordot(S,A,(3,1))
    A = np.tensordot(AS,SA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(4,4,4,4)

    AA = np.tensordot(A,A,(3,1))
    A = np.tensordot(AA,AA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(16,16,16,16)
    
    factor = Trace_tensor(A)
    #factor = 1
    
    A /= factor
    
    return A,factor


def initialize_impurity_site_tensors(T, J2,factor):
    # Make initial tensor of square lattice Ising model at a temperature T
    A =np.zeros((2,2,2,2),dtype=np.float64)
    S =np.zeros((2,2,2,2),dtype=np.float64)
    Te =np.zeros((2,2,2,2),dtype=np.float64)
    Tm =np.zeros((2,2,2,2),dtype=np.float64)
    Tmx =np.zeros((2,2,2,2),dtype=np.float64)
    Tmy =np.zeros((2,2,2,2),dtype=np.float64)

    for i in range(0,2):
        S[i,i,i,i] = 1.0
        si = (i - 0.5) * 2
        for j in range(0,2):
            sj = (j - 0.5) * 2
            for k in range(0,2):
                sk = (k - 0.5) * 2
                for l in range(0,2):
                    sl = (l - 0.5) * 2
                    A[i,j,k,l] = np.exp((0.5 * (si*sj + sj*sk + sk*sl + sl*si) -J2 * (si*sk + sj*sl))/T)
                    Te[i,j,k,l] = -(0.5 * (si*sj + sj*sk + sk*sl + sl*si) -J2 * (si*sk + sj*sl)) * A[i,j,k,l]
                    Tm[i,j,k,l] = 0.25 * (si + sj + sk + sl) * A[i,j,k,l]
                    Tmx[i,j,k,l] = 0.25 * (si + sj - sk - sl) * A[i,j,k,l]
                    Tmy[i,j,k,l] = 0.25 * (si - sj - sk + sl) * A[i,j,k,l]


    AS = np.tensordot(A,S,(3,1))
    TeS = np.tensordot(Te,S,(3,1))
    TmS = np.tensordot(Tm,S,(3,1))
    TmxS = np.tensordot(Tmx,S,(3,1))
    TmyS = np.tensordot(Tmy,S,(3,1))

    
    SA = np.tensordot(S,A,(3,1))
    STe = np.tensordot(S,Te,(3,1))
    STm = np.tensordot(S,Tm,(3,1))
    STmx = np.tensordot(S,Tmx,(3,1))
    STmy = np.tensordot(S,Tmy,(3,1))

    A = np.tensordot(AS,SA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(4,4,4,4)    
    Ae = 0.5 * (np.tensordot(TeS,SA,([2,4],[0,3])) + np.tensordot(AS,STe,([2,4],[0,3]))).transpose(0,2,1,4,5,6,3,7).reshape(4,4,4,4)
    Am = 0.5 * (np.tensordot(TmS,SA,([2,4],[0,3])) + np.tensordot(AS,STm,([2,4],[0,3]))).transpose(0,2,1,4,5,6,3,7).reshape(4,4,4,4)
    Amx = 0.5 * (np.tensordot(TmxS,SA,([2,4],[0,3])) - np.tensordot(AS,STmx,([2,4],[0,3]))).transpose(0,2,1,4,5,6,3,7).reshape(4,4,4,4)
    Amy = 0.5 * (np.tensordot(TmyS,SA,([2,4],[0,3])) + np.tensordot(AS,STmy,([2,4],[0,3]))).transpose(0,2,1,4,5,6,3,7).reshape(4,4,4,4)


    AA = np.tensordot(A,A,(3,1))
    AeA = np.tensordot(Ae,A,(3,1))
    AmA = np.tensordot(Am,A,(3,1))
    AmxA = np.tensordot(Amx,A,(3,1))
    AmyA = np.tensordot(Amy,A,(3,1))

    Ae = np.tensordot(AeA,AA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(16,16,16,16)
    Am = np.tensordot(AmA,AA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(16,16,16,16)
    Amx = np.tensordot(AmxA,AA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(16,16,16,16)
    Amy = np.tensordot(AmyA,AA,([2,4],[0,3])).transpose(0,2,1,4,5,6,3,7).reshape(16,16,16,16)

    ## normalization of tensor

    Ae /= factor
    Am /= factor
    Amx /= factor
    Amy /= factor
    
    return Ae,Am,Amx,Amy
    
    
def Trace_tensor(A):
    # contraction of a single Tensor with the perioic boundary condition
    Trace_A = np.trace(A,axis1=0,axis2=2)
    Trace_A = np.trace(Trace_A)
    return Trace_A

def main():
    ## read params from command line
    args = parse_args()

    T = args.T
    J2 = args.J2
    chi_max = args.chi
    max_itr = args.max_itr
    epsilon = args.epsilon
    lam_epsilon = args.lam_epsilon
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
    initial_type = args.initial_type
    append_flag = args.append

    tag = "_chi"+str(chi_max)
    if append_flag:
        file_output = open("Ising_J1J2_large_output"+tag+".dat","a")
    else:
        file_output = open("Ising_J1J2_large_output"+tag+".dat","w")

    file_output.write("# T = "+repr(T)+"\n")
    file_output.write("# J2 = "+repr(J2)+"\n")
    file_output.write("# chi_max = "+repr(chi_max)+"\n")
    file_output.write("# max_itr = "+repr(max_itr)+"\n")
    file_output.write("# epsilon = "+repr(epsilon)+"\n")
    file_output.write("# lam_epsilon = "+repr(lam_epsilon)+"\n")
    file_output.write("# seed = "+repr(seed)+"\n")
    file_output.write("# initial_type = "+repr(initial_type)+"\n")
    file_output.write("# append_flag = "+repr(append_flag)+"\n")
    file_output.write("## T, f=F/N, m=M/N,   mx=Mx/N,  my=My/N, iteration number, correlation length (along J2 direction)"+"\n")

                        
        
    local_T,factor = initialize_A_site_tensor(T,J2,)
    local_Te,local_Tm,local_Tmx,local_Tmy = initialize_impurity_site_tensors(T,J2,factor)
    
    if initial_type == 1:
        ## mx state
        Tn_init = (np.kron(np.kron(np.array((0,1)),np.array((1,0))),np.kron(np.array((1,0)),np.array((0,1)))) + (np.random.rand(16)-0.5) * 0.01).reshape(1,16,1)
    else:
        Tn_init = (np.random.rand(16)-0.5).reshape(1,16,1)
    lam_init = np.ones(1)

    Tn, lam, eig_val_L = Make_Canonical_form_symmetric(Tn_init,lam_init)

    delta_lam = 1.0
    itr = 0
    while ( delta_lam > epsilon and itr < max_itr):        
        Tn_new,lam_new,truncation_error = iTEBD_MPO1_update(Tn,lam,local_T,chi_max,lam_epsilon,symmetry=False)

        #eig_R, eig_L, vec_R, vec_L = Calc_dominant_vectors(Tn_new,Tn_new,lam_new,lam_new,local_T)
        Tn_D,eig_L_UD = Normalize_two_states(Tn_new,Tn_new.reshape(len(lam_new),2,2,2,2,len(lam_new)).transpose(5,4,3,2,1,0).reshape(len(lam_new),16,len(lam_new)),lam_new,lam_new)        

        eig_R, eig_L, vec_R, vec_L = Calc_dominant_vectors(Tn_new,Tn_D,lam_new,lam_new,local_T)

        norm = Calc_1site(Tn_new,Tn_D,lam_new,lam_new,local_T,vec_R,vec_L)
        ene = Calc_1site(Tn_new,Tn_D,lam_new,lam_new,local_Te,vec_R,vec_L)/norm
        m = Calc_1site(Tn_new,Tn_D,lam_new,lam_new,local_Tm,vec_R,vec_L)/norm
        mx = Calc_1site(Tn_new,Tn_D,lam_new,lam_new,local_Tmx,vec_R,vec_L)/norm
        my = Calc_1site(Tn_new,Tn_D,lam_new,lam_new,local_Tmy,vec_R,vec_L)/norm
        
        ## calc singular value difference to check the convegence
        chi_comp = np.min([len(lam_new),len(lam)])
        delta_lam = np.sqrt(np.sum((lam_new[:chi_comp]-lam[:chi_comp])**2))

        print(repr(itr)+" "  +repr(-0.125*T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(ene))+" " +repr(np.real(m))+" " +repr(np.real(mx))+" " +repr(np.real(my))+" "+repr(delta_lam)+ " "+ repr(truncation_error))
#        print(repr(itr)+" "  +repr(-0.5*T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(ene))+" " +repr(np.real(m))+" " +repr(np.real(mx))+" " +repr(np.real(my))+" "+repr(delta_lam)+ " "+ repr(truncation_error)+" " +repr(eig_R)+" " +repr(eig_L)+ " " +repr(np.dot(vec_R.reshape(-1),vec_L.reshape(-1)))+ " "+repr(np.dot(vec_R.reshape(-1),vec_L.reshape(vec_L.shape[0],2,2,vec_L.shape[0]).transpose(3,2,1,0).reshape(-1))))
        
        Tn = Tn_new
        lam = lam_new
              
        itr += 1

    print("## calculation has finised at "+repr(itr-1)+" steps")
    print(repr(T)+" "  +repr(J2)+ " " +repr(-0.125*T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(ene))+" " +repr(np.real(m))+" " +repr(np.real(mx))+" " +repr(np.real(my))+" "+repr(itr))

    print("## calculate correlation length (along J2 direction)")
    ## calculate correlation length
    xi = Calc_correlation_length(Tn,Tn_D,lam,lam,1)
    print(repr(T)+" "+ repr(xi[0]*2*np.sqrt(2)))
    
    file_output.write(repr(T)+" "  +repr(J2)+ " " +repr(-0.125*T*np.log(np.real(eig_L[0] * factor)))+ " " +repr(np.real(ene))+" " +repr(np.real(m))+" " +repr(np.real(mx))+" " +repr(np.real(my))+" "+repr(itr)+" " + repr(xi[0]*2*np.sqrt(2))+"\n")
                        
if __name__ == "__main__":
    main()

    
