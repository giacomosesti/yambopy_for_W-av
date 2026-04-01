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
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
from netCDF4 import Dataset
from yambopy.kpoints import kfmt,check_kgrid,build_ktree,find_kpt
from yambopy.lattice import rec_lat, car_red, red_car
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.dbs.em1db import YamboScreeningDB 

class QPCheckInterpolateW:
    """
    Class to handle the auxiliary functions of the W-av method from Yambo
    
    This reads the databases ``ndb.RIM_W_aux_func`` and the ``ndb.RIM`` file
       
    It compares the exact calculation with the interpolation/extrapolation applied within the W-av method
 
    """

    def __init__(self,save='.',aux='.',scr_path=None,filename='ndb.RIM_W_aux_func',db1='ns.db1',do_not_read_cutoff=False,freq=None,
                 gr=None,gc=None,fileout=False,Np=51,overwrite=False,out_plot=True):

        self.save = save
        self.aux = aux
        if scr_path==None:
           self.scr_path=aux
        else:
           self.scr_path=scr_path
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
        if gr is None and gc is None:
            range_gr = range(self.ngvectors)
        elif isinstance(gr, int):
            range_gr = [gr]
        elif isinstance(gr, list):
            range_gr = gr  
        else:
            # do the diagonal
            range_gr = [gc] if isinstance(gc, int) else gc 
        
        if isinstance(gc, int):
            range_gc = [gc]
        elif isinstance(gc, list):
            range_gc = gc 
        else:
            range_gc = range_gr
             
        # frequency index setting        
        if isinstance(freq, int):
         range_nfreq = [freq]
        elif isinstance(freq, list):    
         range_nfreq=freq
        else:
         range_nfreq = range(self.f_coeff.shape[0])

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

        #read the screening database
        yem1=YamboScreeningDB(save=self.save,em1=self.scr_path)

        ###################################
        # Get the indexes of the q points #
        ###################################
        
        database_name = "RIMW_check_database.npz"
       
        if not overwrite and os.path.isfile(database_name):
           with np.load(database_name) as data:
            existing_keys = set(data.files) 
            is_missing = False
        
           # Check every expected key
           for iw in range_nfreq:
            for idir in self.i_dirs:
             for igr in range_gr:
              for igc in range_gc:
               kn = f'W_w{iw}_rldir_{idir+1}_num_g{igr}_g{igc}'
               kf = f'W_w{iw}_rldir_{idir+1}_fit_g{igr}_g{igc}'
                        
               if kn not in existing_keys or kf not in existing_keys:
                 print(f" [!] Missing: {kn} or {kf}")
                 is_missing = True
                 break 
              if is_missing: break
             if is_missing: break
            if is_missing: break
        
           if is_missing:
             print(" [!] Database incomplete. Re-calculating...")
             do_calculation = True
           else:
            print(" [IO] Database is complete. Skipping calculation.")
            do_calculation = False
        else:
         do_calculation = True

        if do_calculation: 

         master_database = {}
         for idir in self.i_dirs:
          for igr in range_gr:
           for igc in range_gc:
            for iw in range_nfreq:  

              n_indx_steps=Ngrid[idir] // 2
             
              # define all the quantities used in the loop
              # calculated values
              q_calc=np.zeros(Ngrid[idir])
              v_num=np.zeros(Ngrid[idir])
              f_num=np.zeros(Ngrid[idir],dtype=np.complex64)
              eps_num=np.zeros(Ngrid[idir],dtype=np.complex64)
              W_num=np.zeros(Ngrid[idir],dtype=np.complex64)
              # interpolation
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
               print(' check q g1 g2 ')
    
               indexes=self.get_equiv_index(q_num,self.red_gvectors[igc],self.red_gvectors[igr],ylat)
               print(indexes)
               gr=self.gvectors[igr]*2*np.pi
               gc=self.gvectors[igc]*2*np.pi

               #
               # =================
               #   Reading the calculated values from screening database
               # =================             
               #                 
               vX=yem1.X[indexes[0],iw,indexes[1],indexes[2]]

               #bare Coulomb potential
               q_out=red_car([q_num],self.rlat)[0]*2*np.pi
               q_calc[i]=np.sign(q_num[idir])*np.sqrt(np.dot(q_out,q_out))
               v_num[i]=np.sqrt(v_bare(np.array([q_out]).T,gr,self.n_dirs, self.lcut, self.cut_dir)[0]*v_bare(np.array([q_out]).T,gc,self.n_dirs, self.lcut, self.cut_dir)[0])[0]

               # num value of auxiliary function, eps and W
               eps_num[i]=vX
               W_num[i]=v_num[i]*vX
               f_num[i]=vX/(vX+1)/v_num[i]
               if (igc==igr and iw==0 and  self.rimw_type=='metal'):
                eps_num[i]= 1 + vX
                W_num[i] = (1+vX)*v_num[i]

               #
               # =================
               #   Interpolation/Extrapolation
               # =================             
               #       
               inter=list(range(Np*i,Np*(i+1)))
               #
               # Transform f_coeff from iBZ to BZ
               #
               symi=np.linalg.inv(ylat.sym_red[ylat.symmetry_indexes[bz_index]])
               f_trans=trans_f_coeff(self.f_coeff[iw,indexes[0],indexes[1],indexes[2],:],symi)
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

               # analytic limit for q->0 head and wings
               if((igr==0 or igc==0) and ibz_index==0):
                 for i_dense in range(Np):
                       
                  mem_idx=Np*i+i_dense
                
                  W[mem_idx],epsm1[mem_idx],f_val[mem_idx]= \
                       analytic(ibz_index,self.f_coeff,dq[:,i_dense],q_rlu[:,i_dense],self.red_gvectors,v_coul[mem_idx],vslab[i_dense],
                                iw,igc,igr,(self.n_dirs is 2),self.rimw_type,self.is_anis_on,self.em1_anis,self.rlat,self.alat,idir,self.cut_dir,self.i_dirs) 
               elif(((igr==0 and igc==1) or (igr==1 and igc==0)) and ibz_index==0 and self.rimw_type=='metal'):
                
                 for i_dense in range(Np):

                  mem_idx=Np*i+i_dense

                  W[mem_idx],epsm1[mem_idx],f_val[mem_idx]= \
                       analytic(ibz_index,self.f_coeff,dq[:,i_dense],q_rlu[:,i_dense],self.red_gvectors,v_coul[mem_idx],vslab[i_dense],
                                iw,igc,igr,(self.n_dirs is 2),self.rimw_type,self.is_anis_on,self.em1_anis,self.rlat,self.alat,idir,self.cut_dir,self.i_dirs)
               # general case
               else:
                  f_val[inter] = evaluate_polynomial(dq, f_trans)
                  epsm1[inter] = f_val[inter]*v_coul[inter]/(1.0-v_coul[inter]*f_val[inter])
                  W[inter]= v_coul[inter]*v_coul[inter]*f_val[inter]/(1.0-v_coul[inter]*f_val[inter])
                  if (igc==igr and iw==0 and self.rimw_type == "metal"):
                    epsm1[inter] = 1.0/(1.0-v_coul[inter]*f_val[inter])
                    W[inter]= v_coul[inter]/(1.0-v_coul[inter]*f_val[inter])

              #
              # Store the data, in a in a file or a .npz database
              #
              if fileout==True:
                 filename=f'W_w{iw}_rldir_{idir+1}_num_g{igr}_g{igc}'
                 filename1=f'W_w{iw}_rldir_{idir+1}_fit_g{igr}_g{igc}'
                 fm="18.8e"
                 with open(f"{filename}.dat", "w") as f:
                   f.write("q(bohr^-1)  V_coul(Ha)  Re(f)(Ha^-1)  Im(f)(Ha^-1)  Re(epsm1)"
                            " Im(epsm1)  Re(W)(Ha)  Im(W)(Ha)\n")
                   for j in range(Ngrid[idir]):
                    f.write(f"{q_calc[j]:{fm}}   {v_num[j]:{fm}}   {np.real(f_num[j]):{fm}}   {np.imag(f_num[j]):{fm}}"
                      f"{np.real(eps_num[j]):{fm}}    {np.imag(eps_num[j]):{fm}}   {np.real(W_num[j]):{fm}}   {np.imag(W_num[j]):{fm}}\n")

                 with open(f"{filename1}.dat", "w") as f:
                   f.write("q(bohr^-1)  V_coul(Ha)  Re(f)(Ha^-1)  Im(f)(Ha^-1)  Re(epsm1)"
                            " Im(epsm1)  Re(W)(Ha)  Im(W)(Ha)\n")
                   for j in range(Ngrid[idir]*Np):
                    f.write(f"{q_norm[j]:{fm}}   {v_coul[j]:{fm}}   {np.real(f_val[j]):{fm}}   {np.imag(f_val[j]):{fm}}"
                      f"{np.real(epsm1[j]):{fm}}    {np.imag(epsm1[j]):{fm}}   {np.real(W[j]):{fm}}   {np.imag(W[j]):{fm}}\n")

              else:

               key_num = f'W_w{iw}_rldir_{idir+1}_num_g{igr}_g{igc}'
               key_fit = f'W_w{iw}_rldir_{idir+1}_fit_g{igr}_g{igc}'

               master_database[key_num]= np.column_stack([q_calc,v_num,np.real(f_num),np.imag(f_num), \
                    np.real(eps_num),np.imag(eps_num),np.real(W_num),np.imag(W_num)])

               master_database[key_fit]= np.column_stack([q_norm,v_coul,np.real(f_val),np.imag(f_val), \
                    np.real(epsm1),np.imag(epsm1),np.real(W),np.imag(W)])

         database_name = "RIMW_check_database.npz"
         np.savez_compressed(database_name, **master_database)

         print(f" [IO] RIMW check database dumped")
         print(f" frequencies dumped (by index)",range_nfreq)
         print(f" gvectors dumped (by index) ",range_gr,' and ',range_gc)
         
        # if database available, just declare it is there
        else:

         print(f" [IO] RIMW check database already available")
         print(f" frequencies dumped (by index)",range_nfreq)
         print(f" gvectors dumped (by index) ",range_gr,' and ',range_gc)
     
    def plot_comparison(self,iw=0,ig1=0,ig2=0,directory='rimw_check_figures'):

        # plot the comparison between the numerical data
        # and the interpolated quantities
        #
        # All the data is read from the RIMW_check_database.npz
        #
        # The output figures are stored in directory

        database_name = "RIMW_check_database.npz"
        
        if not os.path.exists(directory):
           os.makedirs(directory)           

        for idir in self.i_dirs: 
         with np.load(database_name) as data:
                 inp_data=data[f'W_w{iw}_rldir_{idir+1}_num_g{ig1}_g{ig2}']
                 inp_file=data[f'W_w{iw}_rldir_{idir+1}_fit_g{ig1}_g{ig2}']

                 #def plot_epsm1(self,ig1=0,ig2=0,**kwargs):
                 figsz=(9,6)
                 fr=[0.0,0.5]
                 string_size = 30
                 colors = np.array(["C1","C2","C3","C4"])
                 widths=[4,0,4,4]
                 markers=['o','P','o','o']
                 markerssz=[10,5,16,8]
                 fcl=['white',"C4",'white',"C6"]
                 pw_exp=[-3,-3,-1,-1,0,0]
                 styles=['solid','solid','solid','solid'] 

                 is_ff_head = (ig1 == 0 and ig2 == 0 )
                 figs,axs = fig_setup(figsz,string_size,ig1,ig2,is_ff_head)
                 if (iw==0):
                   label3=r' $\omega=0$'
                 else:
                   label3=r' $\omega \neq 0$'
                 if (is_ff_head and len(self.i_dirs)==2):
                   axs[4].set_yscale("linear")
                   axs[5].set_yscale("linear")

                 for ncol in range(2,8):
                   if (iw==0 and is_ff_head):
                    if (ncol==4):
                       inp_data[:,ncol]=inp_data[:,4]-1
                       inp_file[:,ncol]=inp_file[:,4]-1
                   
                   xcoords=(inp_data[:-1,0]+inp_data[1:,0])/2
                   inp_data[int((len(inp_data[:,0])+1)/2),ncol]=float("NaN")
                   axs[ncol-2].plot(inp_data[:,0],inp_data[:,ncol],zorder=1,color=colors[0],marker=markers[0],linewidth=0,
                                    markeredgewidth=3,markerfacecolor=fcl[0],markersize=markerssz[0],label=label3) #+label2) #+label3+label2
                   axs[ncol-2].plot(inp_file[:,0],inp_file[:,ncol],zorder=-1,color=colors[0],linewidth=widths[0],linestyle=styles[0],markersize=0)
                 if (iw == 0)  and (ig1==0) and (ig2==0): axs[0].legend(loc='best',handletextpad=0.2,prop={'size': 23},framealpha=1.0)
                 if (iw != 0) and (ig1==0) and (ig2!=0): axs[4].set_ylim(-10,10)
                 if (ig1==0) and (ig2==0): axs[2].set_xlim(-0.46,0.46)
                 ext=[None] * 8
                 a_pos=[0.85]*8
                 if (ig1==0) and (ig2==0) and (iw==0): a_pos[0]=0.1
                 for ncol in range(0,len(axs)):
                  ext[ncol]=axs[ncol].get_ylim() 
                  axs[ncol].set_xlim(-0.6,0.6);
                  axs[ncol].set_ylim(ext[ncol][0],ext[ncol][1])   
                  axs[ncol].annotate(r'$\omega = $'+str(fr[iw])+' Ha',xy=(0.07,a_pos[ncol]),fontsize=23, xycoords='axes fraction',
                          bbox=dict(boxstyle="round",fc=(1.0, 1.0, 1.0),edgecolor='lightgray'))
                  #if (iw == 1) : axs[ncol].legend(loc='best',handletextpad=0.2,prop={'size': 18},title=r'$\omega = $ 0 Ha',framealpha=1.0) 
                  #if (iw != 1) : axs[ncol].legend(loc='best',prop={'size': 18},title=r'$\omega = $'+str(fr[iw-1])+' Ha',framealpha=1.0) 
                  axs[ncol].vlines(xcoords,-10**10,10**12,
                               zorder=-10,linestyle=(0, (5,5)),color='gray',linewidths=1.5)
                  axs[ncol].vlines(-xcoords,-10**10,10**12,
                               zorder=-10,linestyle=(0, (5,5)),color='gray',linewidths=1.5)
                 for ncol in range(2,8):
                  cfr=1
                  axs[ncol-2].tick_params(axis='both',which='minor',width=2.0,length=7.5)
                  if (iw == 1) and (ig1==0) and (ig2!=0): axs[ncol-6].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
                  axs[ncol-2].minorticks_on()
                  if (ncol>6) and (ig1==0) and (ig2==0) and (iw==0): cfr=0
                  if (ncol<4) and (ig1==0) and (ig2!=0): cfr=1
                  if (iw != 1) and (ig1==0) and (ig2==0) and (ncol==8 or ncol==10 or ncol==11): 
                   minor_locator=np.linspace(0,10,21)
                   axs[ncol-2].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(10**minor_locator))
                   axs[ncol-2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                   pass
                  else:  
                   axs[ncol-2].yaxis.set_major_formatter(OOMFormatter(pw_exp[ncol-2], f"%1.{cfr}f"))
                   axs[ncol-2].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
                   axs[ncol-2].yaxis.offsetText.set_fontsize(20)
                 fig_save(figs,idir+1,ig1,ig2,iw)                           
          
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
     indexes=np.zeros(3,dtype=int)
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

def v_bare(q_vec, g_vec, n_dirs=3, lcut=None, cutdir=None):
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

def analytic(iq,f_coeff,q_sampl,q_rlu,gvec,v_coul,vslab,iw,igc,igr,cut_is_slab,rimw_type,is_anis_on,em1_anis,rlat,alat,idir_idx,cutdir=2,idir=3):
   
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
               func = f_coeff[iw,iq,igc,igr,0] + \
                      f_coeff[iw,iq,igc,igr,1] * abs(q_rlu[0]) + \
                      f_coeff[iw,iq,igc,igr,2] * abs(q_rlu[1]) + \
                      f_coeff[iw,iq,igc,igr,3] * abs(q_rlu[2]) + \
                      f_coeff[iw,iq,igc,igr,4] * (q_rlu[0]**2) + \
                      f_coeff[iw,iq,igc,igr,5] * (q_rlu[1]**2) + \
                      f_coeff[iw,iq,igc,igr,6] * (q_rlu[2]**2)
               
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


# graphical stuff

def fig_setup(figsz,string_size,ig1,ig2,is_ff_head):

  #labels
  if (ig1 !=0): 
     if (ig2 !=0): mtx_label  =r"_{\mathbf{G}_{"+str(ig1)+"}\mathbf{G}_{"+str(ig2)+"}}$"
     else:  mtx_label  =r"_{\mathbf{G}_{"+str(ig1)+"}0}$"  
  else:
     if (ig2 !=0):  mtx_label  =r"_{0\mathbf{G}_{"+str(ig2)+"}}$"  
     else:  mtx_label  =r"_{00}$"  
  f_label    =r"$f"            +mtx_label+"] [a.u.]"
  VX_label    =r"$\chi V"         +mtx_label+"]"
  W_label    =r"$W"            +mtx_label+"] [a.u.]"
  ylbl=['Re['+f_label,'Im['+f_label,'Re['+VX_label,'Im['+VX_label,'Re['+W_label,'Im['+W_label]
  
  fig1, ax1  =  plt.subplots(figsize=figsz)
  fig2, ax2  =  plt.subplots(figsize=figsz)
  fig3, ax3  =  plt.subplots(figsize=figsz)
  fig4, ax4  =  plt.subplots(figsize=figsz)
  fig5, ax5  =  plt.subplots(figsize=figsz)
  fig6, ax6  =  plt.subplots(figsize=figsz)

  axs  = [ax1,ax2,ax3,ax4,ax5,ax6]
  figs = [fig1,fig2,fig3,fig4,fig5,fig6]

  for j in range(0,len(axs)):
     axs[j].set_xlabel('q (a.u.)', fontsize = string_size )
     axs[j].set_ylabel(ylbl[j], fontsize = string_size )       
     axs[j].tick_params(labelsize=string_size,width=2.0,length=15)
     for tick in axs[j].xaxis.get_major_ticks():
      tick.label1.set_fontsize(string_size)
     for tick in axs[j].yaxis.get_major_ticks():
      tick.label1.set_fontsize(string_size)
 
  if (is_ff_head):
      axs[4].set_yscale("log")
      axs[5].set_yscale("log")
      #axs[2].set_ylim(-1.0,0)
      #axs[2].set_xlim(0,0.6)

  plt.rcParams['font.size'] = 16

  return figs,axs


def fig_save(figs,direct,ig1,ig2,iw):

    print("Saving figures and closing")

    G1_string = "{0:02d}".format(ig1)
    G2_string = "{0:02d}".format(ig2)
    dir_string = "{0:01d}".format(direct)

    #Save
    figs[0].savefig(f"rimw_check_figures/Re_f-G{G1_string}-G{G2_string}_dir_{dir_string}_iw_{iw}.png",bbox_inches='tight')
    figs[1].savefig(f"rimw_check_figures/Im_f-G{G1_string}-G{G2_string}_dir_{dir_string}_iw_{iw}.png",bbox_inches='tight')
    figs[2].savefig(f"rimw_check_figures/Re_XV-G{G1_string}-G{G2_string}_dir_{dir_string}_iw_{iw}.png",bbox_inches='tight')
    figs[3].savefig(f"rimw_check_figures/Im_XV-G{G1_string}-G{G2_string}_dir_{dir_string}_iw_{iw}.png",bbox_inches='tight')
    figs[4].savefig(f"rimw_check_figures/Re_W-G{G1_string}-G{G2_string}_dir_{dir_string}_iw_{iw}.png",bbox_inches='tight')
    figs[5].savefig(f"rimw_check_figures/Im_W-G{G1_string}-G{G2_string}_dir_{dir_string}_iw_{iw}.png",bbox_inches='tight')

    
    #Close
    for i in range(len(figs)): plt.close(figs[i])


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format
