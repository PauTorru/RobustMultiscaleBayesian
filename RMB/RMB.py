import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sc
import hyperspy.api as hs
import cv2
import sys
import importlib
import os
import espm
import sunsal


def python_neighbour_graph(row,col,I_resol):
    """
    Builds the neighbours array for the RMB algorithm. 

    Parameters:
    ------------
    row, col: int
        number of rows and columns in your image.
    I_resol: list
        List of the multiscales to be used, they have to be odd and listed in increasing value.

    """
    #returns neighbours matrix: each row refers to pixel i ( .ravel() order), neighbours ordered according to I_resol
    # neighbours corresponding to I_resol[i] are out[:,:I_resol[i]**2]
    #only odd values. only ordered from smaller to larger.
    
    local_shifts =[]
    for i,r in enumerate(I_resol):
    
        assert r%2==1
        if i>0:
            assert r>I_resol[i-1] # I_resol has to be ordered by filter size

        r_shifts = (np.meshgrid(range(r),range(r))-np.floor(r/2)).reshape((2,-1)).T.tolist()
        r_tup = [tuple(i) for i in r_shifts]
        local_shifts.extend(list(set(r_tup)-set(local_shifts)))

    
    local_shifts = np.array(local_shifts).astype("int")
    Im = np.array(range(row*col)).reshape(row,col).T
    margin = max(I_resol)//2
    ref_image  = Duplicate_Borders(Im,margin)

    neighbours = np.zeros((row*col,len(local_shifts)))

    for k,n in enumerate(local_shifts):
        neighbours[:,k] = np.roll(ref_image,n,(0,1))[margin:-margin,margin:-margin].ravel()
    if True:
        f = []
        for i in range(row):
            f.append(neighbours[i::row])
        f= np.concatenate(f)
        neighbours = f
    
    return neighbours.astype("int")


def Duplicate_Borders(Y,pad):
    """
    Adds borders to the Y for the RMB calculation.
    Parameters:
    -----------
    Y: array
        Dataset to be padded
    pad: int
        withd of the margin added at each side.
    """
    Y2 = np.pad(Y,pad)[:,:]#,pad:-pad]

  
    Y2[:pad,pad:-pad] = Y2[pad:2*pad,pad:-pad][::-1,:]
    Y2[-pad:,pad:-pad] = Y2[-2*pad:-pad,pad:-pad][::-1,:]
    Y2[:,:pad] = Y2[:,pad:2*pad][:,::-1]
    Y2[:,-pad:] = Y2[:,-2*pad:-pad][:,::-1]

    return Y2



def generate_rscales(data,Gm,I_resol):
    """
    Generates multiscale dataset through FFT convolutions (see generate_rescales_faster(), which is recomended in general)

    Parameters:
    -----------

    data: array
        Spectrum image
    Gm : array
        G matrix generated with espm. It contains the endmembers to be estimated in the columns.
    I_resol: list
        List of the multiscales to be used, they have to be odd and listed in increasing value.


    """
    x,y,e = data.shape
    e,n = Gm.shape
    rd = len(I_resol)
    out = np.zeros((x,y,n,rd))
    
    for i,r in enumerate(I_resol):
        kernel =np.ones((r,r))[:,:,np.newaxis]/r**2
        rscales,_,_,_ = sunsal.sunsal(Gm,fftc(data,kernel,"same").reshape((-1,e)).T,positivity="yes")
        out[:,:,:,i]= rscales.T.reshape((x,y,n))
    out[out<0]=0.
    return out

def generate_rscales_faster(data,Gm,I_resol):
    """
    Generates multiscale dataset through opencv 2Dfilter convultion. It is generally faster unles the spectral dimension is very large.

    Parameters:
    -----------

    data: array
        Spectrum image
    Gm : array
        G matrix generated with espm. It contains the endmembers to be estimated in the columns.
    I_resol: list
        List of the multiscales to be used, they have to be odd and listed in increasing value.


    """
    x,y,e = data.shape
    e,n = Gm.shape
    rd = len(I_resol)
    out = np.zeros((x,y,n,rd))
    
    for i,r in enumerate(I_resol):
        kernel =np.ones((r,r))[:,:,np.newaxis]/r**2
        # filter2d only takes 512 channels at a time
        chunks = int(np.ceil(data.shape[-1]/512))
        convolved =[]
        for c in range(chunks):
            convolved.append( cv2.filter2D(data[:,:,c*512:(c+1)*512],-1,kernel,borderType=cv2.BORDER_REFLECT))

        convolved = np.dstack(convolved)
        
        rscales,_,_,_ = sunsal.sunsal(Gm,convolved.reshape((-1,e)).T,positivity="yes")
        out[:,:,:,i]= rscales.T.reshape((x,y,n))
    out[out<0]=0.
    return out



def create_I_up_bar(multiscale_map,neighbours,I_resol):
    """
    Generates I_up_Bar matrix in the RMB algorithm

    Parameters:
    -----------

    multiscale_map: array
        output of generate_rscales

    neighbours: array

    I_resol: list
        List of the multiscales to be used, they have to be odd and listed in increasing value.




    """

    rows, cols, convs = multiscale_map.shape
    in_mat = multiscale_map.reshape((rows*cols,convs))

    rc,nn = neighbours.shape

    I_up_bar = np.zeros((convs*nn,rc))

    assert rows*cols==rc

    iup = 0
    for r in range(convs):
        for n in range(nn):
            
            I_up_bar[iup,:]=in_mat[neighbours[:,n],r]
            
            iup+=1

    return I_up_bar


def robust_median_bayesian(multiscale_map,neighbours,I_resol,iter_max=2,eps=1e-12,convergence_thershold=1e-3):
    """
    Generates I_up_Bar matrix in the RMB algorithm

    Parameters:
    -----------

    multiscale_map: array
        Multiscales maps of a given endmember.

    neighbours: array

    I_resol: list
        List of the multiscales to be used, they have to be odd and listed in increasing value.

    iter_max: int
        Number of iterations of the RMB algorithm

    eps: float
        Small value greater than zero. Used to avoid NaNs.

    convergence_thershold: float
        RMB algorithm is stop if relative chages in the maps are below this thershold

    Ouputs:
    -------

    Maps: array
        Abundance map of the given endmember
    Uncertainty: array
        Associated uncertainty of the calculated abundance map.
    """
    
    rows, cols, convs = multiscale_map.shape
    rc,nn = neighbours.shape
    
    I_up_bar_original = create_I_up_bar(multiscale_map,neighbours,I_resol)#.reshape((-1,cols,rows)).transpose((0,2,1)).reshape((convs*nn,-1))

    I_up_bar = I_up_bar_original.copy()
    I_up_bar[0,:] = I_up_bar_original[nn,:]
    
    ThreshI = 0.5*np.maximum(0.1,I_up_bar[(convs-1)*nn,:])


    wrs =[]

    for i in range(convs): # distance from its own scale?we'll go with that
        num = abs(I_up_bar[i*nn,:][np.newaxis,:] - I_up_bar[nn*i:nn*(i+1),:])
        denom = 2*ThreshI*I_resol[i]
        wr = np.exp(-(num/denom)**2)
        wrs.append(wr)
        
    WR_scale = np.vstack(wrs)

    WR_scale = np.sqrt(WR_scale)


    NormW = np.maximum(eps,WR_scale.sum(0)) # N
    WR_scale /= NormW
    WRsym_scale = np.minimum(1,WR_scale/np.vstack(convs*[NormW[neighbours].T]))

    iters = 1;

    I_up_bar2 = I_up_bar_original.copy()
    DenomEps = 1+WRsym_scale.shape[0]/2

    Mtot = []
    Mcost = []

    while iters<=iter_max:
        #print("Iteration : {}".format(iters))

        M = (WR_scale*I_up_bar2).sum(0)

        Mtot.append(M)
        vtempp = M#[0,:]
        Mextend = vtempp[neighbours].T
        Mextend = np.maximum(eps,Mextend)

        #update EPS_I

        diffI = (WRsym_scale*(M-I_up_bar2)**2)/2
        Eps_I = np.maximum(eps,np.nansum(diffI,axis=0)/DenomEps)

        vtempp = Eps_I#[0]
        Eps_Iextend = vtempp[neighbours].T

        #update I_Up_Bar2

        for j in range(convs):
            Sig_r = 1./( np.maximum(eps,(WRsym_scale[j*nn:(j+1)*nn]/Eps_Iextend).sum(0)))
            Mu_r = (Mextend*WRsym_scale[j*nn:(j+1)*nn]/Eps_Iextend).sum(0)*Sig_r
            Mu_Sig_r = Mu_r - Sig_r
            Int3 = (Mu_Sig_r + np.sqrt(Mu_Sig_r**2 + 4*Sig_r*I_up_bar[j*nn]))/2

            vtempp = Int3#[0]
            I_up_bar2[j*nn:(j+1)*nn] = vtempp[neighbours].T

        #Check convergence

        if iters>1:
            cost = abs(Mtot[iters-1]-Mtot[iters-2]).sum()/abs(Mtot[iters-2]).sum()
            if cost<convergence_thershold:
                return M.reshape((rows,cols)),np.sqrt(Eps_I.reshape((rows,cols)))
        iters+=1
    return M.reshape((rows,cols)),Eps_I.reshape((rows,cols))


def get_RMB_maps(s,I_resol,G=None,**kwargs):
    """
    Obtain RMB optimized abundace maps of the endmembers in the espm G matrix

    Parameters:
    -----------

    s: EDS_spim
        Specturm image to analyze

    I_resol: list
        List of the multiscales to be used, they have to be odd and listed in increasing value.

    G : Array (E,n_elements)
        Array with the spectra of the endmembers to be estimated.
        If None it is taken from s

    kwargs are passed to the robust median bayesian algorithm

    Ouputs:
    -------

    Maps: array
        RMB optimized abundance maps of all endmembers.

    Variance: array
        Associated variances of the calculated abundance maps.
    """
    if G is None:
        s.build_G()
        if callable(s.G):
            G = s.G()
        else:
            G=s.G

    assert s.data.dtype =="float" 

    print("Generating multiscale")
    rscales = generate_rscales_faster(s.data,G,I_resol)
    print("Done")
    r,c,e = s.data.shape
    print("Calculating Neighbours")
    neighbours = python_neighbour_graph(r,c,I_resol)
    print("Done")
    e,els = G.shape
    maps = []
    eps =[]
    print("BMS calculation")
    for i in range(els):
        print("{} / {}".format(i+1,els))
        m,e = robust_median_bayesian(rscales[:,:,i,:],neighbours,I_resol,**kwargs)
        maps.append(m)
        eps.append(e)
    
    return np.dstack(maps),np.sqrt(np.dstack(eps))


def build_RMB_model(s,maps,G):
    """
    Rebuilds the spectral model of the calculated abundace maps and endmembers.

    Parameters:
    -----------
    s: EDS_spim
        Orgininal dataset 

    maps: array
        abundances from get_RMB_maps

    G: array
        Endmember matrix

    Outputs:
    --------
    EDS_spim
        Model of the dataset.
    """

    r,c,els = maps.shape
    e,els = G.shape
    d = (G@maps.reshape(-1,els).T).reshape((e,r,c))

    out = s.deepcopy()
    out.data = d.transpose([1,2,0])
    
    return out



def atomic_percentage_and_uncertainty_from_RMB_maps(maps,variance):
    """Returns atomic percentage maps and associated error maps from the RMB output. 
    Select the maps and variances of the desired elements

    Parameters
    ----------

    maps: np.array
        Maps of the desired elements. shape is (size x, size y, n_elements)

    variance: np.array
        associated variances to the maps. Same shape

    Returns:
    --------

    quantification: np.array
        In %.

    quantification error: np.array
        standard error of the quantification in %.

    """
    #print("Remember: Don't include background estimations into the maps")

    quant = 100*maps/maps.sum(-1)[:,:,np.newaxis]

    error = 2*np.sqrt(variance)
    q_sum = maps.sum(-1)

    q_sum_error = np.sqrt((error**2).sum(-1))

    q_error = quant*np.sqrt((error/maps)**2+((q_sum_error/q_sum)**2)[:,:,np.newaxis])
    return quant,q_error


    
    