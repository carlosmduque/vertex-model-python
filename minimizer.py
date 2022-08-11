"""Quasistatic solver for vertex models

"""
import numpy as np
import logging
from itertools import count

# from scipy import optimize
# from .. import config
# from ..collisions import auto_collisions
# from ..topology import auto_t1, auto_t3

# from .base import TopologyChangeError, set_pos

from scipy.optimize import minimize, OptimizeResult

from basic_topology import *
from topology_triggers import *

log = logging.getLogger(__name__)

MAX_ITER = 100


class TissueMinimizer:
    """Quasistatic solver performing a gradient descent on a :class:`tyssue.Epithelium`
    object.

    Methods
    -------
    find_energy_min : energy minimization calling `scipy.optimize.minimize`
    approx_grad : uses `optimize.approx_fprime` to compute an approximated
      gradient.
    check_grad : compares the approximated gradient with the one provided
      by the model
    """

    def __init__(self,tissue,ener_function,grad_function,
                    with_t1=False,with_t3=False,
                    minimization_method='CG',gtol=1e-05):
        """Creates a quasistatic gradient descent solver with optional
        type1, type3 and collision detection and solving routines.

        Parameters
        ----------
        with_collisions : bool, default False
            wheter or not to solve collisions
        with_t1 : bool, default False
            whether or not to solve type 1 transitions at each
            iteration.
        with_t3 : bool, default False
            whether or not to solve type 3 transitions
            (i.e. elimnation of small triangular faces) at each
            iteration.

        Those corrections are applied in this order: first the type 1, then the
        type 3, then the collisions

        """
        # self.set_pos = set_pos
        # if with_t1:
        #     self.set_pos = auto_t1(self.set_pos)
        # if with_t3:
        #     self.set_pos = auto_t3(self.set_pos)

        self.tissue = tissue
        self.energy = ener_function
        self.gradient = grad_function
        
        # self.x_1d = self.tissue.vert_df[['x','y']].values.flatten()
        # self.x0_1d = self.x_1d
        
        self.restart = True
        self.rearange = with_t1 or with_t3
        
        self.res = OptimizeResult()
        self.res.success = False
        self.res.message = "Not Started"
        # self.res = {"success": False, "message": "Not Started"}
        self.num_restarts = 0
        self.max_iters = 100 #100
        
        self.method = minimization_method
        self.grad_tol = gtol

    # def find_energy_min(self,**minimize_kw):
               
    #     x0_1d = self.tissue.vert_df[['x','y']].values.flatten()
        
    #     self.res = minimize(self._calc_energy_and_update,x0_1d,
    #                 args=(self.tissue),method='CG',
    #                 jac=self._calc_gradient)
        
        
    #     self.tissue.update_tissue_geometry()

    #     return
    
    def find_energy_min(self,**minimize_kw):
               
        while not self.res.success:
            
            x0_1d = self.tissue.vert_df[['x','y']].values.flatten()
            
            self.res = minimize(self._calc_energy_and_update,x0_1d,
                        args=(self.tissue),method=self.method,
                        jac=self._calc_gradient,
                        options={'gtol': self.grad_tol})
                       
            self.tissue.update_tissue_geometry()
            
            if not self.res.success: 
                collapse_small_faces(self.tissue)
                collapse_short_tissue_edges(self.tissue)
            
                self.num_restarts += 1
                            

            if (self.num_restarts == self.max_iters):
                    break
        return

    def _calc_energy_and_update(self, x_1d, tissue):
        shape = (int(len(x_1d)/2),2)
        vertex_positions = np.reshape(x_1d,shape)
        tissue.vert_df[['x','y']] = vertex_positions
        
        geometry_quantities=['area','perimeter','length']
        tissue.update_tissue_geometry(geometry_quantities=geometry_quantities)
        
        return self.energy(tissue)
    
    def _calc_gradient(self, x_1d, tissue):
        return self.gradient(tissue)
    
    
    
    
# def gradient_descent(gradient, start, learn_rate, tissue, n_iter=50, tolerance=1e-06):
#     vector = start
#     for _ in range(n_iter):
#         diff = -learn_rate * gradient(vector,tissue)
#         if np.all(np.abs(diff) <= tolerance):
#             break
#         vector += diff
#     return vector