import numpy as np

class QPCheckInterpolateW:
    def __init__(self, q, Xw, rim_w_ng, k_grid):
        # Constants and Configuration
        self.np = 51
        self.dq = 1.0 / (self.np - 1)
        self.q_min = 1e-5
        self.gmax = rim_w_ng
        self.lcut=alat(idir(1))/2._SP
        
        # Directions for output file naming
        self.directions = ["1", "2", "3"]
        self.out_unit_w_fit = 200
        self.out_unit_w_num = 400
        
        self.n_indx_steps = k_grid/ 2
        
        # npoints calculation 
        self.npoints = (self.n_indx_steps * 2) + 1
        
        # Allocate indexes: (max_pts, 4, 3) 
        # First 4 indices: iq_ibz, is, ig1, ig2 [cite: 388]
        max_pts = np.max(self.npoints)
        self.indexes = np.zeros((max_pts, 4, 3), dtype=int)

import numpy as np

def evaluate_interpolation_polynomial(q_samples, coeffs):
    """
    Evaluates the 20-term polynomial for a batch of q-points.
    
    Parameters:
    q_samples: np.array of shape (N, 3) - The q-points to evaluate
    coeffs:    np.array of shape (20,) - The f_coeff_loc coefficients
    """
    # Decompose q into components for readability
    q1, q2, q3 = q_samples[:, 0], q_samples[:, 1], q_samples[:, 2]
    
    # Terms 1-4: 
    func = (coeffs[0] + 
            q1 * coeffs[1] + 
            q2 * coeffs[2] + 
            q3 * coeffs[3])
    
    # Terms 5-10:
    func += (q1**2 * coeffs[4] + 2*q1*q2 * coeffs[5] +
             q2**2 * coeffs[7] + 2*q2*q3 * coeffs[8] +
             q3**2 * coeffs[9] + 2*q3*q1 * coeffs[6])
    
    # Terms 11-20: Cubic 
    func += (q1**3 * coeffs[10] + 3*q1**2*q2 * coeffs[11] + 3*q1**2*q3 * coeffs[12] +
             q2**3 * coeffs[14] + 3*q2**2*q3 * coeffs[15] + 3*q2**2*q1 * coeffs[13] +
             q3**3 * coeffs[18] + 3*q3**2*q1 * coeffs[17] + 3*q3**2*q2 * coeffs[16] +
             6*q1*q2*q3 * coeffs[19])
             
    return func

def trans_f_coeff(f_func, symi):
    """
    Complete transformation of 20 coefficients from iBZ to BZ.
    f_func: original 20 coeffs (complex64)
    symi: inverse symmetry matrix (3x3 float)
    """
    f_loc = np.zeros(20, dtype=np.complex64)
    
    # 1. Constant (1 term)
    f_loc[0] = f_func[0]
    
    # 2. Linear (3 terms: 1, 2, 3)
    f_loc[1:4] = np.dot(symi.T, f_func[1:4])
    
    # 3. Quadratic (6 terms: 4, 5, 6, 7, 8, 9)
    # Fortran mapping: 4=q1^2, 5=q1q2, 6=q1q3, 7=q2^2, 8=q2q3, 9=q3^2
    # We construct a temporary 3x3 symmetric matrix to use matrix rotation
    mat_q = np.array([
        [f_func[4], f_func[5], f_func[6]],
        [f_func[5], f_func[7], f_func[8]],
        [f_func[6], f_func[8], f_func[9]]
    ], dtype=np.complex64)
    
    # Rotate: S^T * Mat * S
    mat_q_rot = np.dot(symi.T, np.dot(mat_q, symi))
    
    f_loc[4] = mat_q_rot[0,0]
    f_loc[5] = mat_q_rot[0,1]
    f_loc[6] = mat_q_rot[0,2]
    f_loc[7] = mat_q_rot[1,1]
    f_loc[8] = mat_q_rot[1,2]
    f_loc[9] = mat_q_rot[2,2]

    # 4. Cubic (10 terms: 10-19 in 0-based indexing)
    # This replaces the logic in  [cite: 240-252]
    # We build the full 3x3x3 tensor T_ijk
    T = np.zeros((3, 3, 3), dtype=np.complex64)
    
    # Map the 10 unique Fortran cubic coeffs (11:20) into the 3x3x3 tensor
    # Note: Fortran 11-20 -> Python 10-19
    # q1^3, q1^2q2, q1^2q3, q2^2q1, q2^3, q2^2q3, q3^2q1, q3^2q2, q3^3, q1q2q3
    T[0,0,0] = f_func[10] # q1^3
    T[0,0,1] = T[0,1,0] = T[1,0,0] = f_func[11] # q1^2q2
    T[0,0,2] = T[0,2,0] = T[2,0,0] = f_func[12] # q1^2q3
    T[1,1,0] = T[1,0,1] = T[0,1,1] = f_func[13] # q2^2q1
    T[1,1,1] = f_func[14] # q2^3
    T[1,1,2] = T[1,2,1] = T[2,1,1] = f_func[15] # q2^2q3
    T[2,2,0] = T[2,0,2] = T[0,2,2] = f_func[16] # q3^2q1
    T[2,2,1] = T[2,1,2] = T[1,2,2] = f_func[17] # q3^2q2
    T[2,2,2] = f_func[18] # q3^3
    # The q1q2q3 term has a factor of 6 in the polynomial, 
    # so we distribute it across the 6 permutations in the tensor
    T[0,1,2] = T[0,2,1] = T[1,0,2] = T[1,2,0] = T[2,0,1] = T[2,1,0] = f_func[19]

    # Perform the tensor rotation: T'_ijk = S_ia S_jb S_kc T_abc
    # Using np.einsum is the most efficient way to do this in Python
    T_rot = np.einsum('ia,jb,kc,abc->ijk', symi, symi, symi, T)

    # Map the rotated tensor back to the 10 coefficients
    f_loc[10] = T_rot[0,0,0]
    f_loc[11] = T_rot[0,0,1]
    f_loc[12] = T_rot[0,0,2]
    f_loc[13] = T_rot[1,1,0]
    f_loc[14] = T_rot[1,1,1]
    f_loc[15] = T_rot[1,1,2]
    f_loc[16] = T_rot[2,2,0]
    f_loc[17] = T_rot[2,2,1]
    f_loc[18] = T_rot[2,2,2]
    f_loc[19] = T_rot[0,1,2]

    return f_loc

def get_v_bare_general(q_vec, is_slab, idir_idx, alat):
    """
    General bare propagator logic.
    q_vec: [q1, q2, q3]
    """
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-9: return 0.0

    if is_slab:
        v_bare = (2.0 * np.pi * alat[idir_idx]) / q_norm
        return v_bare
    else:
        # 3D Bulk: 4*pi / q^2
        return (4.0 * np.pi) / (q_norm**2)


# 1. Open your database
ds = nc.Dataset('ndb.RIM_W_aux_func', 'r')
f_coeff_data = ds.variables['RIM_W_aux_func_coeff'] # shape (n_freq, n_q_ibz, n_g, n_g, 20)
rimw_type = str(ds.variables['RIM_W_type'][:]).strip().lower()
em1_anis = ds.variables['X_anis_coeff'][:]

anis_str = "".join(ds.variables['Anisotropy'][:]).strip().lower()
is_anis_on = (anis_str == "true")




# 2. Get the specific symmetry matrix (example for index 'is')
# symi is the inverse of rl_sop at that symmetry index
symi = np.linalg.inv(rl_sop[is_idx])

# 3. Pull coefficients for specific indices and transform
ibz_coeffs = f_coeff_data[i_freq, i_q, i_g1, i_g2, :]
transformed_coeffs = trans_f_coeff(ibz_coeffs, symi)


import numpy as np

def get_single_w_check(iw, igr, igc, idir_idx, 
                       f_coeff_all, vx_all, bare_qpg, 
                       q_map, delta_q, n_indx_steps, npts_dir,
                       rimw_type, is_slab, rim_id_ref, em1_anis):
    """
    Evaluates Numerical W for specific frequency (iw) and G-pair (igr, igc).
    
    Parameters:
    iw, igr, igc: Indices provided by the user.
    f_coeff_all: From NetCDF (n_freq, n_q, n_g, n_g, 20)
    vx_all:      From NetCDF (n_freq, n_q, n_g, n_g)
    q_map:       A dictionary/function mapping q-vectors (tuple) to index 'iq'.
    idir_idx:    0, 1, or 2 (Direction X, Y, or Z).
    """
    
    results = []
    npts = npts_dir[idir_idx]
    
    # Pre-fetch system type and reference flags
    # rim_id_ref is assumed to be boolean (True/False)
    
    for i in range(npts):
        # 1. Calculate q-vector for this step
        q_num = delta_q[idir_idx] * (i - n_indx_steps[idir_idx])
        q_norm = np.linalg.norm(q_num)
        
        # 2. Get the index 'iq' using your map
        # Assuming q_map is a dict with rounded q-tuples as keys
        q_key = tuple(np.round(q_num, 6))
        iq = q_map.get(q_key, None)
        
        if iq is None:
            continue # Skip if q is outside the database

        # 3. Numerical Evaluation (Step 3 in Fortran)
        # Fetch bare potentials and vX
        v_bare_igr = bare_qpg[iq, igr]
        v_bare_igc = bare_qpg[iq, igc]
        
        vslab_num = (4.0 * np.pi) / (v_bare_igr * v_bare_igc) # Simplification

        # Pull numerical vX (from NetCDF)
        vx_val = vx_all[iw, iq, igr, igc]
        
        # Numerical f: func_num = (v_bare_g1 * v_bare_g2 / 4pi) * (vX / (1 + vX))
        f_num = (v_bare_igr * v_bare_igc / (4.0 * np.pi)) * (vx_val / (1.0 + vx_val))

        epsm1_num = vx_val
        
        # Handle the specialized rimw_type logic
        if rimw_type.lower() == 'metal':
            epsm1_num = vx_val + 1    
        
        # W calculation
        w_num = epsm1_num * vslab_num

        # Store for the user
        results.append({
            'q_norm': q_norm,
            'f_real': f_num.real,
            'f_imag': f_num.imag,
            'epsm1_real': epsm1_num.real,
            'epsm1_imag': epsm1_num.imag,
            'w_real': w_num.real,
            'w_imag': w_num.imag
        })

    return results



# Constants and temporary variables assumed available:
# pi, alat, q_sampl, q_out_norm, q_rlu, f_coeff, iomega, 
# cut_is_slab, rimw_type, idir, is_anis_on, em1_anis, vslab, r1

if iq == 0:  # Fortran iq=1 is Python 0
    # ---------------------------------------------------------
    # HEAD (G1 = 0, G2 = 0)
    # ---------------------------------------------------------
    if ig1 == 0 and ig2 == 0:

        if rimw_type == "metal":
            if iw == 0: 
                q_out = q_sampl
                # c2a call logic (coordinate conversion) should happen here
                
                # Polynomial evaluation for Metal Head
                func = f_coeff[iw,iq,ig1,ig2,0] + \
                       f_coeff[iw,iq,ig1,ig2,1] * abs(q_rlu[0]) + \
                       f_coeff[iw,iq,ig1,ig2,2] * abs(q_rlu[1]) + \
                       f_coeff[iw,iq,ig1,ig2,3] * abs(q_rlu[2]) + \
                       f_coeff[iw,iq,ig1,ig2,4] * (q_rlu[0]**2) + \
                       f_coeff[iw,iq,ig1,ig2,5] * (q_rlu[1]**2) + \
                       f_coeff[iw,iq,ig1,ig2,6] * (q_rlu[2]**2)
                
                epsm1_sampl = 1.0/(1.0 - (vslab/r1)*func)
                W_sampl = (vslab/r1)/(1.0 - (vslab/r1)*func)
                
                # Limit Check
                if iku_v_norm(q_sampl) < q0_def_norm:
                    if cut_is_slab:
                        epsm1_sampl = q_out_norm/(q_out_norm-2.0*pi*func*alat[idir[0]])
                        W_sampl = 2.0*pi*a_mat[idir[0],idir[0]]/(q_out_norm-2.0*pi*func*alat[idir[0]])
                    else:
                        epsm1_sampl = q_out_norm**2/(q_out_norm**2 - 4.0*pi*func)
                        W_sampl = 4.0*pi/(q_out_norm**2 - 4.0*pi*func)
                # Output/Cycle logic here...

            else: # Metal Finite Frequency
                func = f_coeff[iw,0,0,0,0]
                
                if iku_v_norm(q_sampl) < q0_def_norm:
                    epsm1_sampl = vslab * func / (1.0 - vslab * func)
                    W_sampl = (func / (1.0 - vslab * func)) * (vslab**2 / q0_def_norm**2)
                    func = f_coeff[iw,0,0,0,0] * iku_v_norm(q_sampl)**2
                    # Output/Cycle logic here...
                
                # Finite frequency exponential form
                sign_val = np.sign(f_coeff[iw,0,0,0,2].real)
                sqrt_val = np.sqrt(f_coeff[iw,0,0,0,1]**2 * q_rlu[0]**2 + 
                                   f_coeff[iw,0,0,0,2]**2 * q_rlu[1]**2 + 
                                   f_coeff[iw,0,0,0,3]**2 * q_rlu[2]**2)
                
                func = (func + f_coeff[iw,0,0,0,4]*abs(q_rlu[0]) + 
                               f_coeff[iw,0,0,0,5]*abs(q_rlu[1]) + 
                               f_coeff[iw,0,0,0,6]*abs(q_rlu[2])) * np.exp(sign_val * sqrt_val)
                
                func = func*iku_v_norm(q_sampl)**2
                epsm1_sampl = (func*vslab/r1) / (1.0 - (vslab/r1)*func)
                W_sampl = (vslab/r1) * (func*vslab/r1) / (1.0 - (vslab/r1)*func)

        # ---------------------------------------------------------
        # Semiconductors and Dirac Cone
        # ---------------------------------------------------------
        if iku_v_norm(q_sampl) < 1e-5:
            epsm1_sampl = vslab*f_coeff[iw,0,0,0,0]/(1.0-vslab*f_coeff[iw,0,0,0,0])
            W_sampl = (f_coeff[iw,0,0,0,0]/(1.0-vslab*f_coeff[iw,0,0,0,0]))*(vslab**2/q0_def_norm**2)
            
            if iw == 0 and rimw_type_str == "Dirac":
                epsm1_sampl = (vslab / q0_def_norm * f_coeff[iw,0,0,0,0]) / \
                              (1.0 - vslab / q0_def_norm * f_coeff[iw,0,0,0,0])
                W_sampl = (f_coeff[iw,0,0,0,0]/(1.0-vslab/q0_def_norm*f_coeff[iw,0,0,0,0])) * \
                          (vslab**2 / q0_def_norm**3)

            if is_anis_on and iw == 0:
                W_sampl *= em1_anis[idir_idx]
            # Output/Cycle...

        # Analytical Exponential Form for Semiconductors
        if cut_is_slab:
            sign_val = np.sign(f_coeff[iw,0,0,0,1].real)
            sqrt_val = np.sqrt(f_coeff[iw,0,0,0,1]**2 * q_rlu[0]**2 + 
                               f_coeff[iw,0,0,0,2]**2 * q_rlu[1]**2)
            func = f_coeff[iw,0,0,0,1] * np.exp(sign_val * sqrt_val)
        else:
            sign_val = np.sign(f_coeff[iw,0,0,0,1].real)
            sqrt_val = np.sqrt(f_coeff[iw,0,0,0,1]**2 * q_rlu[0]**2 + 
                               f_coeff[iw,0,0,0,2]**2 * q_rlu[1]**2 + 
                               f_coeff[iw,0,0,0,3]**2 * q_rlu[2]**2)
            func = f_coeff[iw,0,0,0,0] * np.exp(sign_val * sqrt_val)

        if is_anis_on and iw == 0:
            anis_fact = np.dot(em1_anis, (2.0*pi*q_sampl/alat)**2) / iku_v_norm(q_sampl)**2
            func *= anis_fact

        # Final scaling based on type
        if rimw_type in ["semiconductor","Dirac"]:
            power = 2 if rimw_type == "semiconductor" else 1
            func *= iku_v_norm(q_sampl)**power
        else:
            # Default polynomial head (simplified for brevity, follow your f_coeff_loc mapping)
            pass

    # ---------------------------------------------------------
    # WINGS (G1 = 0, G2 != 0)
    # ---------------------------------------------------------
    if ig1 == 0 and ig2 != 0:
        if rimw_type == "metal":
           if cut_is_slab:
             if iw == 0:
                    func = f_coeff[iw,iq,0,ig2,1].real * abs(q_rlu[0]) + \
                           f_coeff[iw,iq,0,ig2,2].real * abs(q_rlu[1]) + \
                           f_coeff[iw,iq,0,ig2,3].real * abs(q_rlu[2]) + \
                           f_coeff[iw,iq,0,ig2,1].imag * q_rlu[0] + \
                           f_coeff[iw,iq,0,ig2,2].imag * q_rlu[1] + \
                           f_coeff[iw,iq,0,ig2,3].imag * q_rlu[2] + \
                           f_coeff[iw,iq,0,ig2,4] * (q_rlu[0]**2) + \
                           f_coeff[iw,iq,0,ig2,5] * (q_rlu[1]**2) + \
                           f_coeff[iw,iq,0,ig2,6] * (q_rlu[2]**2) + \
                           f_coeff[iw,iq,0,ig2,7] * np.sign(q_rlu[0]**2,q_rlu[0]) + \
                           f_coeff[iw,iq,0,ig2,8] * np.sign(q_rlu[1]**2,q_rlu[1]) + \
                           f_coeff[iw,iq,0,ig2,9] * np.sign(q_rlu[2]**2,q_rlu[2]))
             else:
                # Finite Frequency Wings
                # Check for specific G-vector directions (idir 2 and 3)
                is_g_zero = (g_vec[ig2, idir[1]] == g_vec[ig2, idir[2]]) and (g_vec[ig2, idir[1]] == 0)
                
                # Cubic expansion with parity (sign) checks
                term1 = (f_coeff[iw,iq,0,ig2,1] + f_coeff[iw,iq,0,ig2,4] + 
                        (f_coeff[iw,iq,0,ig2,1] - f_coeff[iw,iq,0,ig2,4])*np.sign(1.0,q_rlu[0]))
                term2 = (f_coeff[iw,iq,0,ig2,2] + f_coeff[iw,iq,0,ig2,5] + 
                        (f_coeff[iw,iq,0,ig2,2] - f_coeff[iw,iq,0,ig2,5])*np.sign(1.0,q_rlu[1]))
                term3 = (f_coeff[iw,iq,0,ig2,3] + f_coeff[iw,iq,0,ig2,6] +
                        (f_coeff[iw,iq,0,ig2,3] - f_coeff[iw,iq,0,ig2,6])*np.sign(1.0,q_rlu[2]))
                # Cubic terms c8-c13
                term4 = (f_coeff[iw,iq,0,ig2,7] + f_coeff[iw,iq,0,ig2,10] + 
                        (f_coeff[iw,iq,0,ig2,7] - f_coeff[iw,iq,0,ig2,10])*np.sign(1.0, q_rlu[0]))
                term5 = (f_coeff[iw,iq,0,ig2,8] + f_coeff[iw,iq,0,ig2,11] + 
                        (f_coeff[iw,iq,0,ig2,8] - f_coeff[iw,iq,0,ig2,11])*np.sign(1.0, q_rlu[1]))
                term6 = (f_coeff[iw,iq,0,ig2,9] + f_coeff[iw,iq,0,ig2,12] + 
                        (f_coeff[iw,iq,0,ig2,9] - f_coeff[iw,iq,0,ig2,12])*np.sign(1.0, q_rlu[2]))
                if is_g_zero:
                    func = (term1*q_rlu[0]**2 + term2*q_rlu[1]**2 + term3*q_rlu[2]**2 + term4*q_rlu[0]**3 + term5*q_rlu[1]**3 + term6*q_rlu[2]**3) / 2.0
                else:
                    func = (term1*q_rlu[0] + term2*q_rlu[1] + term3*q_rlu[2] + term4*q_rlu[0]**2 + term5*q_rlu[1]**2 + term6*q_rlu[2]**2) / 2.0
           # 
           # ---------------------------------------------------------
           # LIMIT CHECK FOR WINGS (Analytical regularisation)
           # ---------------------------------------------------------
           # 
           if iku_v_norm(q_sampl) < q0_def_norm:
            if cut_is_slab:
              if iw == 0:
                # System 2D: f func behaves as fcoeff * q / norm(q)
                func_num = (f_coeff[iw,iq,0,ig2,1].real*abs(q_rlu[0]) + 
                            f_coeff[iw,iq,0,ig2,2].real*abs(q_rlu[1]) + 
                            f_coeff[iw,iq,0,ig2,3].real*abs(q_rlu[2]) + 
                            f_coeff[iw,iq,0,ig2,1].imag*q_rlu[0] + 
                            f_coeff[iw,iq,0,ig2,2].imag*q_rlu[1] + 
                            f_coeff[iw,iq,0,ig2,3].imag*q_rlu[2])
                func_val = func_num / iku_v_norm(q_sampl)
                
                # W_sampl and epsm1 scaling for the Wing limit
                W_sampl = (func_val/iku_v_norm(qpGc + q_sampl)**2) *(vslab**2 / q0_def_norm)
                epsm1_sampl = (func_val/iku_v_norm(qpGc + q_sampl)) *vslab*np.sqrt(iku_v_norm(q_sampl)/q0_def_norm)
                func = func_val*iku_v_norm(q_sampl)
              else:
                # Finite frequency wings near limit
                if (iw != 0) and is_g_zero:
                    W_sampl, epsm1_sampl, func = 0.0, 0.0, 0.0
                else:
                    func_val = ((f_coeff[iw,iq,0,ig2,1] -f_coeff[iw,iq,0,ig2,4])*abs(q_rlu[0]) + 
                                (f_coeff[iw,iq,0,ig2,2] -f_coeff[iw,iq,0,ig2,5])*abs(q_rlu[1]) + 
                                (f_coeff[iw,iq,0,ig2,3] -f_coeff[iw,iq,0,ig2,6])*abs(q_rlu[2])) / (iku_v_norm(q_sampl) * 2.0)
                    W_sampl = (func_val / iku_v_norm(qpGc + q_sampl)**2) * (vslab**2 / q0_def_norm)
                    epsm1_sampl = (func_val / iku_v_norm(qpGc + q_sampl)) * vslab * np.sqrt(iku_v_norm(q_sampl) / q0_def_norm)
                    func = func_val * iku_v_norm(q_sampl)
            else:
               # System 3D: Static f func behaves as fcoeff * q^2 / norm(q)^2
               func_val = (q_sampl[0]*(q_sampl[0]*f_coeff_loc[4] + 2.0*q_sampl[1]*f_coeff_loc[5]) + 
                           q_sampl[1]*(q_sampl[1]*f_coeff_loc[7] + 2.0*q_sampl[2]*f_coeff_loc[8]) + 
                           q_sampl[2]*(q_sampl[2]*f_coeff_loc[9] + 2.0*q_sampl[0]*f_coeff_loc[6])) / iku_v_norm(q_sampl)**2
            
               W_sampl = (func_val / iku_v_norm(qpGc + q_sampl)**2) * (vslab**2)
               epsm1_sampl = (func_val / iku_v_norm(qpGc + q_sampl)) * vslab * iku_v_norm(q_sampl)
               func = func_val * iku_v_norm(q_sampl)**2





