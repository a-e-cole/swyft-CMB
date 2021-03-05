from classy import Class
import healpy as hp
import numpy as np
import swyft
import torch
import math
from swyft.nn import OnlineNormalizationLayer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#prior = swyft.Prior({"h": ['uniform', 0.4,1.0]})
prior = swyft.Prior({"h": ['uniform',0.6685,0.6715]})
NSIDE=128


#### now define simulator ###################################
# Define your cosmology (what is not specified will be set to CLASS default parameters)




# Run the whole code. Depending on your output, it will call the
# CLASS modules more or less fast. For instance, without any
# output asked, CLASS will only compute background quantities,
# thus running almost instantaneously.
# This is equivalent to the beginning of the `main` routine of CLASS,
# with all the struct_init() methods called.
#############################################

def simulator(h, sigma=0.1):
    # Create an instance of the CLASS wrapper
    cosmo = Class()

    # Set the parameters to the cosmological code
    p = {
    'output': 'tCl lCl',
    'l_max_scalars': 500,
    'lensing': 'yes',
    'A_s': 2.3e-9,
    'n_s': 0.9624, 
    'h': h,
    'omega_b': 0.022068,
    'omega_cdm': 0.12029}
    cosmo.set(p)
    cosmo.compute()

    # Access the lensed cl until l=500
    cls = cosmo.lensed_cl(384)
    cosmo.struct_cleanup()
    #x=hp.sphtfunc.synfast(cls['tt'], nside=NSIDE)

    #return x
    #ells=[l*(l+1) for l in range(501)]
    
    #return cls['tt']*ells
    #ls=np.array([0]+[0]+[h]+[0]*407)
    #return hp.sphtfunc.synfast(h*np.arange(501)+sigma*np.random.rand(501)+1, nside=NSIDE)
    return cls['tt'][:384]
    #cls=hp.sphtfunc.anafast(theMap, nside=NSIDE)
    #return hp.
    #return [h]*196608+sigma*np.random.rand(196608)

def model(params):
    """Model wrapper around simulator code."""
    mu = simulator(params["h"])
    return dict(mu=mu)

ell=np.array([l for l in range(0,384)])
ells=np.array([l*(l+1) for l in range(0,384)])
def noise(obs, params=None, sigma=1.0):
    """Associated noise model. For Cl's, cosmic variance."""
    # multiply by l(l+1) for normalization?
    # fidCl*np.sqrt(2/(2*ells+1))[3:]
    Cl=obs['mu']
    sigma=(Cl*np.sqrt(2/(2*ell+1)))
    Cl=np.multiply(np.random.rand(*Cl.shape),sigma)+Cl
    return {'mu': Cl*ells*10E9}
    #data = {k: hp.sphtfunc.anafast(hp.sphtfunc.synfast(v,nside=NSIDE))*ells*10e9 for k, v in obs.items()}
    #return data

'''from deepsphere.layers.samplings.healpix_pool_unpool import Healpix
from deepsphere.layers.samplings.healpix_pool_unpool import HealpixMaxUnpool as hpmup
from deepsphere.layers.samplings.healpix_pool_unpool import HealpixAvgPool as hpap
from deepsphere.layers.samplings.healpix_pool_unpool import HealpixAvgUnpool as hpaup
from deepsphere.utils.laplacian_funcs import get_healpix_laplacians
from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.layers.chebyshev import ChebConv'''

import torch.nn.functional as F
#bugfix for HealpixMaxPool function
'''class HealpixMaxPool(torch.nn.MaxPool1d):
    """Healpix Maxpooling module
    """

    def __init__(self, return_indices=False):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=return_indices)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch

        Args:
            x (:obj:`torch.tensor`):[batch x pixels x features]

        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [batch x pooled pixels x features] and indices of pooled pixels
        """
        x = x.permute(0, 2, 1)
        if self.return_indices:
            x, indices = F.max_pool1d(x, self.kernel_size)
        else:
            x = F.max_pool1d(x,self.kernel_size)
        x = x.permute(0, 2, 1)

        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output
    
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from deepsphere.utils.laplacian_funcs import prepare_laplacian

from deepsphere.utils.samplings import (
    healpix_resolution_calculator
)
def get_healpix_laplacians(nodes, depth, laplacian_type):
    """Get the healpix laplacian list for a certain depth.
    Args:
        nodes (int): initial number of nodes.
        depth (int): the depth of the UNet.
        laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
    Returns:
        laps (list): increasing list of laplacians.
    """
    laps = []
    for i in range(depth):
        pixel_num = nodes
        resolution = int(healpix_resolution_calculator(pixel_num)/2**i)
        G = SphereHealpix(nside=resolution, n_neighbors=8)
        G.compute_laplacian(laplacian_type)
        laplacian = prepare_laplacian(G.L)
        laps.append(laplacian)
    return laps[::-1]'''

'''laps=get_healpix_laplacians(12*NSIDE**2,7,"normalized")
llaps1=laps[-1].cuda()
llaps2=laps[-2].cuda()
llaps3=laps[-3].cuda()
llaps4=laps[-4].cuda()
llaps5=laps[-5].cuda()
llaps6=laps[-6].cuda()
llaps7=laps[-7].cuda()'''

'''class CustomHead(swyft.Module):
    def __init__(self, obs_shapes):
        super().__init__(obs_shapes=obs_shapes)

        self.n_features = 12*640

        #self.spch1 = SphericalChebConv(1, 10, laps[-1], kernel_size=3)
        #self.spch2 = SphericalChebConv(10, 20, laps[-2], kernel_size=3)
        #self.spch3 = SphericalChebConv(20, 40, laps[-3], kernel_size=3)
        self.spch1 = ChebConv(1, 10, kernel_size=5)
        self.spch2 = ChebConv(10, 20, kernel_size=5)
        self.spch3 = ChebConv(20, 40, kernel_size=5)
        self.spch4 = ChebConv(40, 80, kernel_size=5)
        self.spch5 = ChebConv(80, 160, kernel_size=5)
        self.spch6 = ChebConv(160, 320, kernel_size=5)
        self.spch7 = ChebConv(320, 640, kernel_size=5)
        self.pool = HealpixMaxPool()
        #self.l = torch.nn.Linear(192*40, 1000)
        self.onl_f = OnlineNormalizationLayer(torch.Size([196608]))

    def forward(self, obs):
        x = obs["mu"]
        nbatch = len(x)
        #x = torch.log(0.1+x)
        x=self.onl_f(x)
        x=x.unsqueeze(-1)

        x = self.spch1(llaps1,x)
        x = self.pool(x)
        x = self.spch2(llaps2,x)
        x = self.pool(x)
        x = self.spch3(llaps3,x)
        x = self.pool(x)
        x = self.spch4(llaps4,x)
        x = self.pool(x)
        x = self.spch5(llaps5,x)
        x = self.pool(x)
        x = self.spch6(llaps6,x)
        x = self.pool(x)
        x = self.spch7(llaps7,x)
        x = self.pool(x)
        x=torch.softmax(x,dim=1)
        x = x.view(nbatch, -1)
        #x = self.l(x)

        return x'''
    
class HeadCl(swyft.Module):
    def __init__(self, obs_shapes):
        super().__init__(obs_shapes=obs_shapes)

        self.n_features = 40*46

        self.conv1 = torch.nn.Conv1d(1, 10, 3)
        self.conv2 = torch.nn.Conv1d(10, 20, 3)
        self.conv3 = torch.nn.Conv1d(20, 40, 3)
        self.pool = torch.nn.MaxPool1d(2)
        #self.l = torch.nn.Linear(40, 10)
        #self.onl_f = OnlineNormalizationLayer(torch.Size([384]))
        self.onl_f = OnlineNormalizationLayer(torch.Size([384]))
        

    def forward(self, obs):
        x = obs["mu"]
        nbatch = len(x)
        #x = torch.log(0.1+x)
        x=self.onl_f(x)
        x=x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pool(x)
        x=F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x=F.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x=F.relu(x)
        x = x.view(nbatch, -1)
        #x = self.l(x)

        return x


#The data is not generated by our simulator; par0 is unknown.
par0 = {'h': 0.67}
obs0=model(par0)
