# classical_iTEBD
python implementations of iTEBD for classical square lattice Ising models.
* Main algorithms
  * iTEBD_classical.py 
    * Based on R. Orus and G. Vidal Phys. Rev. B 78, 155117 􏰀(2008􏰁). I modified the definition of singualr values in Fig.13 (algorith for 2-site iMPS). (It might be a typo...).
* Square lattice Ising model
  * iTEBD_Ising.py
    * iTEBD with 1-site iMPS. 
* Square lattice J1-J2 Ising model
  * local tensor is defined as T_0 in H. Li and L.-P. Yang, Phys. Rev. B 104, 024118 (2021). T_0 is 4-leg tensor with dimension (4,4,4,4).
  * iTEBD_J1J2_Ising.py
    * Assuming 1-site iMPS. It is not suitable for the stripe phase.
  * iTEBD_J1J2_Ising_large.py
    * Assuming 1-site iMPS and the local tensor is the 2x2 unit of T_0. This can simulate the stripe phase with mx order.
  * iTEBD_J1J2_Ising_2site.py
    * Assuming 2-site iMPS with the local tensor T_0. This can simulate the stripe phase with both of mx and my orders.

# Usage
We need numpy and scipy. Input parameters for each application will be shown, e.g., 

``` python3 iTEBD_Ising.py -h  ```

The output contains, 

* free energy density
* energy density
* uniform magnetization
* stripe magnetization (mx)
* stripe magnetization (my)

(The definition of mx and my are in PRB 104, 024118 (2021).)
