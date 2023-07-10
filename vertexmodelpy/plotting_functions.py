import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection,LineCollection

import numpy as np


def plot_ellipse(ellipse_coefficients,x_boundary,y_boundary,num_points=[300,300]):
    
    x_coord = np.linspace(x_boundary[0],x_boundary[1],num_points[0])
    y_coord = np.linspace(y_boundary[0],y_boundary[1],num_points[1])
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    
    Z_coord = ellipse_coefficients[0] * X_coord ** 2 + \
            2*ellipse_coefficients[1] * X_coord * Y_coord + \
            ellipse_coefficients[2] * Y_coord**2 + \
            2*ellipse_coefficients[3] * X_coord + \
                2*ellipse_coefficients[4] * Y_coord

    ellip_plot = plt.contour(X_coord,Y_coord,Z_coord,levels=[1],colors=('r'),linewidths=2)
    
    return ellip_plot