#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: GS
#
# This file is part of the yambopy project
#
import numpy as np
import os
from netCDF4 import Dataset
from yambopy.kpoints import kfmt,check_kgrid,build_ktree,find_kpt
from yambopy.lattice import rec_lat, car_red, red_car
from yambopy.dbs.latticedb import YamboLatticeDB

class QPCheckInterpolateW:
    """
    Class to handle the auxiliary functions of the W-av method from Yambo
    
    This reads the databases ``ndb.RIM_W_aux_func`` and the ``ndb.RIM`` file
       
    It compares the exact calculation with the interpolation/extrapolation applied within the W-av method
 
    """

    def __init__(self,save='.',aux='.',filename='ndb.RIM_W_aux_func',db1='ns.db1',do_not_read_cutoff=False,freq=None,gr=None,gc=None,fileout=False,Np=51):

        self.save = save
        self.aux = aux
        self.filename = filename
        self.no_cutoff = do_not_read_cutoff
        
        #read lattice parameters
        if os.path.isfile('%s/%s'%(self.save,db1)):
            try:
                database = Dataset("%s/%s"%(self.save,db1), 'r')
                self.alat = database.variables['LATTICE_PARAMETER'][:]
                self.lat  = database.variables['LATTICE_VECTORS'][:].T
                gvectors = database.variables['G-VECTORS'][:].T
                self.volume = np.linalg.det(self.lat)
                self.rlat = rec_lat(self.lat)      

                #read q-points
                self.iku_q = database.variables['K-POINTS'][:].T
                self.car_q = np.array([ q/self.alat for q in self.iku_q ]) #atomic units
                self.red_q = car_red(self.car_q,self.rlat) 
                self.nqpoints = len(self.car_q)

            except:
                raise IOError("Error opening %s."%db1)
        else:
            raise FileNotFoundError("File %s not found."%db1)      

        #read RIM_W_aux_func database
        if not os.path.isfile("%s/%s"%(self.aux,self.filename)): 
           raise FileNotFoundError("File %s not found."%self.filename)
        else:
          try:
            database = Dataset("%s/%s"%(self.aux,self.filename), 'r')
          except:
            raise IOError("Error opening %s/%s in RIM_W auxiliary function database"%(self.save,self.filename))

        ylat = YamboLatticeDB.from_db_file(filename='SAVE/ns.db1',Expand=True)
        ktree = build_ktree(ylat.red_kpoints)
        self.ktree = ktree
        self.kmap         = ylat.BZ_to_IBZ_indexes
        self.inv_kmap     = ylat.IBZ_to_BZ_indexes
        self.full         = ylat.iku_kpoints

        #read coefficients of the auxiliary functions       
        f_temp = database.variables['RIM_W_aux_func_coeff'] # complex, shape (n_freq, n_q_ibz, n_g, n_g, 20, 2) 
        self.f_coeff=f_temp[...,0]+1j*f_temp[...,1]

        # keep only the g vectors for which the RIM_W is done
        self.ngvectors = self.f_coeff.shape[3]
        # g vectors indexes setting
        if (gr is None and gc is None):
         range_gr = range(self.ngvectors)
         range_gc = range_gr
        else:
         if gr is not None:
             r_idx = gr
         else:
             r_idx = gc # Default to gc if gr is missing for diagonal
             
         # Determine gc index
         if gc is not None:
             c_idx = gc
         else:
             c_idx = r_idx # Default to gr if gc is missing for diagonal
             
         range_gr = [r_idx]
         range_gc = [c_idx]

        # frequency index setting        
        if freq==None:
         range_nfreq = range(self.f_coeff.shape[0])
        else:
         range_nfreq=[freq]

        self.gvectors = np.array([ g/self.alat for g in gvectors[:self.ngvectors] ])      
        self.red_gvectors = car_red(self.gvectors,self.rlat)

        # read interpolation type
        self.rimw_type = b"".join(database.variables['RIM_W_type'][:]).decode('utf-8').strip().lower()
        
        #check anisotropy
        self.em1_anis = database.variables['X_anis_coeff'][:]
        anis_str = b"".join(database.variables['Anisotropy'][:]).decode('utf-8').strip().lower()
        self.is_anis_on = (anis_str == "true")

        #check if cutoff present and supported
        self.cutoff = b"".join(database.variables['CUTOFF'][:]).decode('utf-8').strip().lower()
        self.Np=Np
        distq = 1./(self.Np-1)
        
        dirs=[0,1,2]
        print(anis_str)
        supported_cutoffs = ['none','slab x','slab y','slab z']
        if self.cutoff not in supported_cutoffs: raise NotImplementedError("[ERROR] The W-av method is not currently implemented with this cutoff %s."%self.cutoff)
        if self.cutoff in supported_cutoffs[1:]:
              idx=supported_cutoffs[1:].index(self.cutoff) 
              self.lcut=self.alat[idx]/2
              self.n_dirs  = 2
              self.i_dirs=[ dirs[j] for j in dirs if j!= idx ]
              self.cut_dir = idx
        else:
             self.n_dirs = 3
             self.i_dirs=dirs

        deltaq_rlu,deltaq_iku= self.get_deltak()
        print(deltaq_iku)
        print(deltaq_rlu)
          
        Ngrid, min_dk_rlu=check_kgrid(self.red_q,self.rlat)

        ###################################
        # Get the indexes of the q points #
        ###################################
        
        for idir in self.i_dirs:
         for igr in range_gr:
          for igc in range_gc:
           for iw in range_nfreq:  
              n_indx_steps=Ngrid[idir] // 2
             
              # define all the quantities used in the loop
              # Np is the sampling in each miniBZ
              q_norm=np.zeros(Ngrid[idir]*Np)
              v_coul=np.zeros(Ngrid[idir]*Np) 
              f_val=np.zeros(Ngrid[idir]*Np,dtype=np.complex64)
              epsm1=np.zeros(Ngrid[idir]*Np,dtype=np.complex64)
              W=np.zeros(Ngrid[idir]*Np,dtype=np.complex64)

              for i in range(0,Ngrid[idir]):         
            
               q_num = np.zeros(3)
               q_num[idir] = deltaq_rlu[idir]*(i-n_indx_steps)
               bz_index = find_kpt(self.ktree,q_num)
               ibz_index = self.kmap[bz_index]
               print(' check q g1 g2')
    
               indexes=self.get_equiv_index(q_num,self.red_gvectors[igr],self.red_gvectors[igc],ylat)
               print(indexes)
               inter=list(range(Np*i,Np*(i+1)))
               #
               # Transform f_coeff from iBZ to BZ
               #
               symi=np.linalg.inv(ylat.sym_red[ylat.symmetry_indexes[bz_index]])
               f_trans=trans_f_coeff(self.f_coeff[iw,np.int64(indexes[0]),np.int64(indexes[2]),np.int64(indexes[1]),:],symi)
               #
               # we define the q_sampl for which we calculate the interpolated quantities
               # the q_sampl are always centered in 0, only the interpolated quantities depend on the bz_index of input
               #
               q_sampl = np.zeros((3,Np))
               q_sampl[idir,:] = (distq*(np.arange(Np)-1)-0.5)*deltaq_rlu[idir]
               #
               # keep q_sampl in rlu
               #
               q_rlu=q_sampl
               #
               # Convert q_sampl,gc and gr to iku
               #
               dq=(red_car(q_rlu.T,self.rlat)*self.alat).T
               #
               # Save total q,G, G' and total norm of q in cc 
               #
               q_out = red_car((q_num[:, None]+q_rlu).T,self.rlat).T*2*np.pi
               q_norm[inter]= np.sign(q_num[idir,None]+q_rlu[idir,:])*np.linalg.norm(q_out,axis=0)          
               gr=self.gvectors[igr]*2*np.pi
               gc=self.gvectors[igc]*2*np.pi
               #
               # =================
               #   Evaluate v_col
               # =================             
               #
               vr,vslabr =v_bare(q_out,gr,self.n_dirs, self.lcut, self.cut_dir)
               vc,vslabc =v_bare(q_out,gc,self.n_dirs, self.lcut, self.cut_dir)
               v_coul[inter]=np.sqrt(vr*vc)
               vslab=np.sqrt(vslabr*vslabc)
               #
               # evaluate the interpolating polynomial
               #
               #
               if((igr==0 or igc==0) and ibz_index==0):
                 for i_dense in range(Np):
                       
                  mem_idx=Np*i+i_dense
                
                  W[mem_idx],epsm1[mem_idx],f_val[mem_idx]= \
                       analytic(ibz_index,self.f_coeff,dq[:,i_dense],q_rlu[:,i_dense],self.red_gvectors,v_coul[mem_idx],vslab[i_dense],
                                iw,igr,igc,(self.n_dirs is 2),self.rimw_type,self.is_anis_on,self.em1_anis,self.rlat,self.alat,idir,self.cut_dir,self.i_dirs) 
               elif(((igr==0 and igc==1) or (igr==1 and igc==0)) and ibz_index==0 and self.rimw_type=='metal'):
                
                 for i_dense in range(Np):

                  mem_idx=Np*i+i_dense

                  W[mem_idx],epsm1[mem_idx],f_val[mem_idx]= \
                       analytic(ibz_index,self.f_coeff,dq[:,i_dense],q_rlu[:,i_dense],self.red_gvectors,v_coul[mem_idx],vslab[i_dense],
                                iw,igr,igc,(self.n_dirs is 2),self.rimw_type,self.is_anis_on,self.em1_anis,self.rlat,self.alat,idir,self.cut_dir,self.i_dirs)
                 
               else:
                  f_val[inter] = evaluate_polynomial(dq, f_trans)
                  epsm1[inter] = f_val[inter]*v_coul[inter]/(1.0-v_coul[inter]*f_val[inter])
                  W[inter]= v_coul[inter]*v_coul[inter]*f_val[inter]/(1.0-v_coul[inter]*f_val[inter])
                  if (igc==igr and iw==0 and self.rimw_type == "metal"):
                    epsm1[inter] = 1.0/(1.0-v_coul[inter]*f_val[inter])
                    W[inter]= v_coul[inter]/(1.0-v_coul[inter]*f_val[inter])
              #
              # Store the data, in a .npz database or in a file
              #
              filename=f'W_w{iw}_rldir_{idir+1}_fit_g{igr}_g{igc}'
              if fileout==True:
                 fm="18.8e"
                 with open(f"{filename}.dat", "w") as f:
                   f.write("q(bohr^-1)  V_coul(Ha)  Re(f)(Ha^-1)  Im(f)(Ha^-1)  Re(epsm1)" 
                            " Im(epsm1)  Re(W)(Ha)  Im(W)(Ha)\n")
                   for j in range(Ngrid[idir]*Np):
                    f.write(f"{q_norm[j]:{fm}}   {v_coul[j]:{fm}}   {np.real(f_val[j]):{fm}}   {np.imag(f_val[j]):{fm}}"
                      f"{np.real(epsm1[j]):{fm}}    {np.imag(epsm1[j]):{fm}}   {np.real(W[j]):{fm}}   {np.imag(W[j]):{fm}}\n")
              else:
                 filename = f"{filename}.npz"
              
                 np.savez(f"{filename}",a=q_norm,b=v_coul,c=np.real(f_val),d=np.imag(f_val),
                    e=np.real(epsm1),f=np.imag(epsm1),g=np.real(W),h=np.imag(W))
                 print(" [IO] {filename} file dumped")


    def get_deltak(self):
        kx = self.red_q[:,0]
        ky = self.red_q[:,1]
        kz = self.red_q[:,2]

        def ind_min_pos(a,tol=1e-5):
          return np.where(a>tol,a,np.inf).argmin()

        idxs = [ind_min_pos(kx), ind_min_pos(ky), ind_min_pos(kz)]
        
        delta_rlu = np.zeros(3)
        delta_iku = np.zeros((3, 3))
             
        for i, idx in enumerate(idxs):
            delta_rlu[i] = self.red_q[idx,i]
            delta_iku[i, :] = self.iku_q[idx, :]
            if abs(delta_rlu[i])<1e-6: 
               delta_rlu[i]=1
               delta_iku[i,i]= delta_rlu[i]

        return delta_rlu, delta_iku

    def get_equiv_index(self,q_target,G1_target,G2_target,ylat):
       
     # Assuming G1_target and G2_target are the RLU G-vectors you are checking
     indexes=np.zeros(3)
     qpGr = q_target + G1_target
     qpGc = q_target + G2_target
     find = 0
     # Nested loops to find the triple (iq_ibz, is, ig) that reconstructs the targets
     for ig_trial in range(self.ngvectors):
         g_vec = self.red_gvectors[ig_trial]
          
         for iq_bz in range(len(self.full)):
          iq_ibz = self.kmap[iq_bz]
          S = ylat.sym_red[np.where(self.inv_kmap[iq_ibz]==iq_bz)]
          # Apply symmetry: S * (q_ibz + G_trial)
          qpG_trial = np.matmul(S,self.red_q[iq_ibz] + g_vec)
  
          # Match G1
          if np.allclose(qpGr, qpG_trial, atol=1e-5):
             indexes[0] = iq_ibz
             indexes[1] = ig_trial # ig1
             find += 1
              
          # Match G2
          if np.allclose(qpGc, qpG_trial, atol=1e-5):
             indexes[2] = ig_trial # ig2
             find += 1
              
          if find == 2: break
         if find == 2: break
     if (find != 2): raise SystemExit("not found the matching q+G and q+G'")      
     return indexes   
 

def evaluate_polynomial(dq, coeffs):
#   
#   Evaluates the 20-term polynomial for a batch of q-points.
#   
#   dq: (3,Np) array of displacements
#   coeffs: (20,) array of transformed coefficients
#   
    x, y, z = dq[0,:], dq[1,:], dq[2,:]
    # Construct a matrix of shapes (20, Np)
    terms = np.array([
        np.ones_like(x), x, y, z,             # constant/linear
        x**2, x*y, x*z, y**2, y*z, z**2,      # quadratic
        x**3, x**2*y, x**2*z, y**2*x, y**3,   # cubic
        y**2*z, z**2*x, z**2*y, z**3, x*y*z   # 
    ])
    return np.dot(coeffs, terms)

def trans_f_coeff(f_func, symi):
#   """
#   Complete transformation of 20 coefficients from iBZ to BZ.
#   f_func: original 20 coeffs (complex64)
#   symi: inverse symmetry matrix (3x3 float)
#   """
   f_loc = np.zeros(20, dtype=np.complex64)
   
   # Constant (1 term)
   f_loc[0] = f_func[0]
   
   # Linear (3 terms: 1, 2, 3)
   f_loc[1:4] = np.dot(symi.T, f_func[1:4])
   
   # Quadratic (6 terms: 4, 5, 6, 7, 8, 9)
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

   #  Cubic (10 terms: 10-19 in 0-based indexing) 
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

def v_bare(q_vec, g_vec, n_dirs, lcut, cutdir):
    """
    Calculates the bare Coulomb potential v_G(q).
    q_vec: q-point in Cartesian (Bohr^-1)
    g_vec: G-vector in Cartesian (Bohr^-1)
    lcut:  cutoff distance (alat[idx]/2)
    """
    # Total momentum Q = q + G
    q0_def_norm=1e-12
    Q = q_vec + g_vec[:,None]
    q_norm_sq = np.sum(Q**2, axis=0)
    # in nd out of plane components 
    if n_dirs == 2:
        # --- 2D Slab (here named for Z-cutoff, logic all-wise) ---
        q_z = Q[cutdir,:]
        # q_xy norm of the in-plane components, q_z out-of plane
        q_in_sq = q_norm_sq - q_z**2
        q_xy = np.sqrt(np.maximum(q_in_sq, 0.0))
    # Ensure denominator is never 0.0 to prevent the warning
    denom = np.where(q_norm_sq < q0_def_norm, q0_def_norm**2, q_norm_sq)
    v_pre = (4.0 * np.pi / denom)

    if n_dirs == 2:
        q_xy=np.where(q_xy < q0_def_norm, q0_def_norm, q_xy)
        term_cutoff = 1.0 - np.exp(-q_xy * lcut) * np.cos(np.abs(q_z) * lcut)
        v_val = v_pre * term_cutoff
    else:
        v_val = v_pre
        v_val=np.where(q_norm_sq < 1e-12,4*np.pi/q0_def_norm**2, v_val)
    
    vslab=v_val*denom
    return v_val,vslab

def analytic(iq,f_coeff,q_sampl,q_rlu,gvec,v_coul,vslab,iw,igr,igc,cut_is_slab,rimw_type,is_anis_on,em1_anis,rlat,alat,idir_idx,cutdir=2,idir=3):
   
   q0_def_norm=1e-12
   q_out = red_car([q_rlu],rlat)[0]*2*np.pi
   q_out_norm=np.sqrt(np.dot(q_out,q_out))
   # ---------------------------------------------------------
   # HEAD (G1 = 0, G2 = 0)
   # ---------------------------------------------------------
   if igr == 0 and igc == 0:

       if rimw_type == "metal":
           if iw == 0: 
 
               # Polynomial evaluation for Metal Head
               func = f_coeff[iw,iq,igr,igc,0] + \
                      f_coeff[iw,iq,igr,igc,1] * abs(q_rlu[0]) + \
                      f_coeff[iw,iq,igr,igc,2] * abs(q_rlu[1]) + \
                      f_coeff[iw,iq,igr,igc,3] * abs(q_rlu[2]) + \
                      f_coeff[iw,iq,igr,igc,4] * (q_rlu[0]**2) + \
                      f_coeff[iw,iq,igr,igc,5] * (q_rlu[1]**2) + \
                      f_coeff[iw,iq,igr,igc,6] * (q_rlu[2]**2)
               
               epsm1_sampl = 1.0/(1.0 - v_coul*func)
               W_sampl = v_coul/(1.0 - v_coul*func)
               
               # Limit Check
               if q_out_norm < q0_def_norm:
                   if cut_is_slab:
                       epsm1_sampl = q_out_norm/(q_out_norm-2.0*np.pi*func*alat[cutdir])
                       W_sampl = 2.0*np.pi*alat[cutdir]/(q_out_norm-2.0*np.pi*func*alat[cutdir])
                   else:
                       epsm1_sampl = q_out_norm**2/(q_out_norm**2 - 4.0*np.pi*func)
                       W_sampl = 4.0*np.pi/(q_out_norm**2 - 4.0*np.pi*func)
                   return W_sampl,epsm1_sampl,func
               return W_sampl,epsm1_sampl,func 
#
           else: # Metal Finite Frequency
               func = f_coeff[iw,0,0,0,0]
               
               if q_out_norm < q0_def_norm:
                   epsm1_sampl = vslab * func / (1.0 - vslab * func)
                   W_sampl = (func / (1.0 - vslab * func)) * (vslab**2 / q0_def_norm**2)
                   func = f_coeff[iw,0,0,0,0] * np.dot(q_sampl,q_sampl)
                   return W_sampl,epsm1_sampl,func              

               # Finite frequency exponential form
               sign_val = np.sign(f_coeff[iw,0,0,0,2].real)
               sqrt_val = np.sqrt(f_coeff[iw,0,0,0,1]**2 * q_rlu[0]**2 + 
                                  f_coeff[iw,0,0,0,2]**2 * q_rlu[1]**2 + 
                                  f_coeff[iw,0,0,0,3]**2 * q_rlu[2]**2)
               
               func = (func + f_coeff[iw,0,0,0,4]*abs(q_rlu[0]) + 
                              f_coeff[iw,0,0,0,5]*abs(q_rlu[1]) + 
                              f_coeff[iw,0,0,0,6]*abs(q_rlu[2])) * np.exp(sign_val * sqrt_val)
               
               func = func*q_out_norm**2
               epsm1_sampl = func*v_coul / (1.0 - v_coul*func)
               W_sampl = v_coul * func*v_coul / (1.0 - v_coul*func)
               return W_sampl,epsm1_sampl,func

       # ---------------------------------------------------------
       # Semiconductors and Dirac Cone
       # ---------------------------------------------------------

       if q_out_norm < 1e-5:
           epsm1_sampl = vslab*f_coeff[iw,0,0,0,0]/(1.0-vslab*f_coeff[iw,0,0,0,0])
           W_sampl = (f_coeff[iw,0,0,0,0]/(1.0-vslab*f_coeff[iw,0,0,0,0]))*(vslab**2/q0_def_norm**2)
           
           if iw == 0 and rimw_type_str == "Dirac":
               epsm1_sampl = (vslab / q0_def_norm * f_coeff[iw,0,0,0,0]) / \
                             (1.0 - vslab / q0_def_norm * f_coeff[iw,0,0,0,0])
               W_sampl = (f_coeff[iw,0,0,0,0]/(1.0-vslab/q0_def_norm*f_coeff[iw,0,0,0,0])) * \
                         (vslab**2 / q0_def_norm**3)

           if is_anis_on and iw == 0:
               W_sampl *= em1_anis[idir_idx]
           return W_sampl,epsm1_sampl,func

       # Analytical Exponential Form for Semiconductors
       if cut_is_slab:
           sign_val = np.sign(f_coeff[iw,0,0,0,1].real)
           sqrt_val = np.sqrt(f_coeff[iw,0,0,0,1]**2 * q_rlu[0]**2 + 
                              f_coeff[iw,0,0,0,2]**2 * q_rlu[1]**2)
           func = f_coeff[iw,0,0,0,0] * np.exp(sign_val * sqrt_val)
       else:
           sign_val = np.sign(f_coeff[iw,0,0,0,1].real)
           sqrt_val = np.sqrt(f_coeff[iw,0,0,0,1]**2 * q_rlu[0]**2 + 
                              f_coeff[iw,0,0,0,2]**2 * q_rlu[1]**2 + 
                              f_coeff[iw,0,0,0,3]**2 * q_rlu[2]**2)
           func = f_coeff[iw,0,0,0,0] * np.exp(sign_val * sqrt_val)

       if is_anis_on and iw == 0:
           anis_fact = np.dot(em1_anis, (2.0*np.pi*q_sampl/alat)**2) / q_out_norm
           func *= anis_fact

#       # Final scaling based on type
       if rimw_type in ["semiconductor","Dirac"]:
           power = 2 if rimw_type == "semiconductor" else 1
           func *= q_out_norm**power
           return W_sampl,epsm1_sampl,func
       else:
           # Default polynomial head (simplified for brevity, follow your f_coeff_loc mapping)
           pass
           return W_sampl,epsm1_sampl,func

   # ---------------------------------------------------------
   # WINGS (G1 = 0, G2 != 0)
   # ---------------------------------------------------------
   if (igr == 0 and igc != 0) or (igr != 0 and igc == 0):
       if rimw_type == "metal":
          if cut_is_slab:
            if iw == 0:
                   func = f_coeff[iw,iq,igc,igr,1].real * abs(q_rlu[0]) + \
                          f_coeff[iw,iq,igc,igr,2].real * abs(q_rlu[1]) + \
                          f_coeff[iw,iq,igc,igr,3].real * abs(q_rlu[2]) + \
                          1j*f_coeff[iw,iq,igc,igr,1].imag * q_rlu[0] + \
                          1j*f_coeff[iw,iq,igc,igr,2].imag * q_rlu[1] + \
                          1j*f_coeff[iw,iq,igc,igr,3].imag * q_rlu[2] + \
                          f_coeff[iw,iq,igc,igr,4].real * (q_rlu[0]**2) + \
                          f_coeff[iw,iq,igc,igr,5].real * (q_rlu[1]**2) + \
                          f_coeff[iw,iq,igc,igr,6].real * (q_rlu[2]**2) + \
                          1j*f_coeff[iw,iq,igc,igr,4].imag * np.sign(q_rlu[0])*(q_rlu[0]**2) + \
                          1j*f_coeff[iw,iq,igc,igr,5].imag * np.sign(q_rlu[1])*(q_rlu[1]**2) + \
                          1j*f_coeff[iw,iq,igc,igr,6].imag * np.sign(q_rlu[2])*(q_rlu[2]**2) + \
                          f_coeff[iw,iq,igc,igr,7] * np.sign(q_rlu[0])*(q_rlu[0]**2) + \
                          f_coeff[iw,iq,igc,igr,8] * np.sign(q_rlu[1])*(q_rlu[1]**2) + \
                          f_coeff[iw,iq,igc,igr,9] * np.sign(q_rlu[2])*(q_rlu[2]**2)
            else:
               # Finite Frequency Wings
               # Check for specific G-vector directions (idir 2 and 3)
               is_g_zero = (gvec[igc, idir[0]] == gvec[igc, idir[1]]) and (gvec[igc, idir[0]] == 0)
               
               # Cubic expansion with parity (sign) checks
               term1 = (f_coeff[iw,iq,igc,igr,1] + f_coeff[iw,iq,igc,igr,4] + 
                       (f_coeff[iw,iq,igc,igr,1] - f_coeff[iw,iq,igc,igr,4])*np.sign(q_rlu[0]))
               term2 = (f_coeff[iw,iq,igc,igr,2] + f_coeff[iw,iq,igc,igr,5] + 
                       (f_coeff[iw,iq,igc,igr,2] - f_coeff[iw,iq,igc,igr,5])*np.sign(q_rlu[1]))
               term3 = (f_coeff[iw,iq,igc,igr,3] + f_coeff[iw,iq,igc,igr,6] +
                       (f_coeff[iw,iq,igc,igr,3] - f_coeff[iw,iq,igc,igr,6])*np.sign(q_rlu[2]))
               # Cubic terms c8-c13
               term4 = (f_coeff[iw,iq,igc,igr,7] + f_coeff[iw,iq,igc,igr,10] + 
                       (f_coeff[iw,iq,igc,igr,7] - f_coeff[iw,iq,igc,igr,10])*np.sign(q_rlu[0]))
               term5 = (f_coeff[iw,iq,igc,igr,8] + f_coeff[iw,iq,igc,igr,11] + 
                       (f_coeff[iw,iq,igc,igr,8] - f_coeff[iw,iq,igc,igr,11])*np.sign(q_rlu[1]))
               term6 = (f_coeff[iw,iq,igc,igr,9] + f_coeff[iw,iq,igc,igr,12] + 
                       (f_coeff[iw,iq,igc,igr,9] - f_coeff[iw,iq,igc,igr,12])*np.sign(q_rlu[2]))
               if is_g_zero:
                   func = (term1*q_rlu[0]**2 + term2*q_rlu[1]**2 + term3*q_rlu[2]**2 + term4*q_rlu[0]**3 + term5*q_rlu[1]**3 + term6*q_rlu[2]**3) / 2.0
               else:
                   func = (term1*q_rlu[0] + term2*q_rlu[1] + term3*q_rlu[2] + term4*q_rlu[0]**2 + term5*q_rlu[1]**2 + term6*q_rlu[2]**2) / 2.0
          # 
          # ---------------------------------------------------------
          # LIMIT CHECK FOR WINGS (Analytical regularisation)
          # ---------------------------------------------------------
          # 
          if q_out_norm < q0_def_norm:
           #qpG
           if (igr != 0):
              qpG = red_car([gvec[igr]],rlat)[0]*2*np.pi
           else:
              qpG = red_car([gvec[igc]],rlat)[0]*2*np.pi
           qpG_norm=np.sqrt(np.dot(qpG,qpG))
           if cut_is_slab:
            # the auxliary func are in rlu, we take the limit diving by q in cc, a rlu_to_cc factor has to be placed
            fact_cc_rlu=np.zeros(3)
            for j in idir:
             dummy_rlu=np.zeros(3)
             dummy_rlu[j]=1e-3
             fact_cc_rlu[j]=1e-3/(2*np.pi*np.sqrt(np.dot(red_car([dummy_rlu],rlat)[0],red_car([dummy_rlu],rlat)[0])))
            #
            if iw == 0:              
               # System 2D: f func behaves as fcoeff * q / norm(q)
               func_val = (f_coeff[iw,iq,igc,igr,1].real*fact_cc_rlu[0] + 
                           f_coeff[iw,iq,igc,igr,2].real*fact_cc_rlu[1] +
                           f_coeff[iw,iq,igc,igr,3].real*fact_cc_rlu[2])/2

               # W_sampl and epsm1 scaling for the Wing limit
               W_sampl = (func_val/qpG_norm**2)*vslab**2/q0_def_norm
               epsm1_sampl = (func_val/qpG_norm)*vslab
               func = func_val*q_out_norm
            else:
               # Finite frequency wings near limit
               if (iw != 0) and is_g_zero:
                   W_sampl, epsm1_sampl, func = 0.0, 0.0, 0.0
               else:
                   func_val = ((f_coeff[iw,iq,igc,igr,1] -f_coeff[iw,iq,igc,igr,4])*fact_cc_rlu[0] + 
                               (f_coeff[iw,iq,igc,igr,2] -f_coeff[iw,iq,igc,igr,5])*fact_cc_rlu[1] + 
                               (f_coeff[iw,iq,igc,igr,3] -f_coeff[iw,iq,igc,igr,6])*fact_cc_rlu[2]) / (2.0)
                   W_sampl = (func_val / qpG_norm**2) * (vslab**2 / q0_def_norm)
                   epsm1_sampl = (func_val / qpG_norm) * vslab * q_out_norm / q0_def_norm
                   func = func_val * q_out_norm
           else:
              # define dummy rlu vector to take the limit, q_sampl in iku, q_out in cc
              dummy_rlu=np.zeros(3)   
              dummy_rlu[idir_idx]=1e-3
              q_out = red_car([dummy_rlu],rlat)[0]*2*np.pi
              q_out_norm=np.sqrt(np.dot(q_out,q_out))
              q_sampl=(red_car([dummy_rlu],rlat)*alat)
              # System 3D: Static f func behaves as fcoeff * q^2
              # Dinamically, f func has a linear term that however cancels out due to symmetry
              #
              func_val = (q_sampl[0]*(q_sampl[0]*f_coeff_loc[4] + 2.0*q_sampl[1]*f_coeff_loc[5]) + 
                          q_sampl[1]*(q_sampl[1]*f_coeff_loc[7] + 2.0*q_sampl[2]*f_coeff_loc[8]) + 
                          q_sampl[2]*(q_sampl[2]*f_coeff_loc[9] + 2.0*q_sampl[0]*f_coeff_loc[6])) / q_out_norm**2
           
              W_sampl = (func_val / qpG_norm**2) * (vslab**2)
              epsm1_sampl = (func_val / qpG_norm) * vslab * q_out_norm
              func = func_val * q_out_norm**2
           return W_sampl,epsm1_sampl,func

       epsm1_sampl = func*v_coul/(1.0-v_coul*func)
       W_sampl= v_coul*v_coul*func/(1.0-v_coul*func)
       return W_sampl,epsm1_sampl,func 




