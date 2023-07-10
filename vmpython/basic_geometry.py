import numpy as np
import math
import pandas as pd
# from utilities import pad_array_2d

def euclidean_distance(point1,point2,multiple_distance=False):
    """ calculates Euclidean distance between two points
        in dimension d.

        Definition:
        -----------
        distance = \sqrt{(x_1-x_2)**2 + (y_1-y_2)**2)}
        where x1 and x2 (y1 and y2) are the x (y) coordinates
        of the two points. 
        
        Parameters:
        -----------
        point1 : Numpy.ndarray, shape: (d,)
            Coordinates first point.
        point2 : Numpy.ndarray, shape: (d,)
            Coordinates second point.

        Returns:
        -----------
        _ : float
            Euclidean distance (norm) of point1 - point2.
    """
    
    if multiple_distance:
        return np.linalg.norm(point1-point2,axis=1)
    else:
        return np.linalg.norm(point1-point2)
    # return np.linalg.norm(point1-point2)

def pol_perimeter(vert_positions):
    """ Calculates the perimeter of a polygon by finding the Euclidean
        distance between consecutive vertex positions.

        Definition:
        -----------
        length = \sum_i \sqrt{(x_{i+1}-x_i)**2 + (y_{i+1}-y_i)**2)}
        where x_i and y_i are respectively the ordered x and y coordinates
        from the different polygon's corners. 
        
        Parameters:
        -----------
        vert_positions : Numpy.ndarray, shape: (N,2)
            Ordered vertex position of the polygon.

        Returns:
        -----------
        length : float
            Polygon perimeter.
    """

    # length = [euclidean_distance(v1,v2) for v1,v2 in
    #         zip(vert_positions,np.roll(vert_positions,1, axis=0))]
    
    vert_positions_roll = np.roll(vert_positions.astype(float), 1, axis=0)
    length = euclidean_distance(vert_positions.astype(float),
                vert_positions_roll, multiple_distance=True)

    length = np.sum(length)

    return length

def pol_area(vert_positions):
    """ Uses the Shoelace formula to calculate area of polygon with
        N sides. Positive (negative) for anticlokwise (clokwise) ordered
        polygon vertices.

        Definition:
        -----------
        area = (1/2)\sum_i (x_i y_{i+1} - x_{i+1} y_i)

        where x_i and y_i are respectively the ordered x and y coordinates
        from the different polygon's corners. 
        
        Parameters:
        -----------
        vert_positions : Numpy.ndarray, shape: (N,2)
            Ordered vertex position of the polygon.

        Returns:
        -----------
        area : float
            Signed polygon area.
    """
    x, y = vert_positions[:,0], vert_positions[:,1]

    # area = np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    # area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    # area *= 0.5
    determinant_array = x * np.roll(y,-1) - np.roll(x,-1) * y
    area = 0.5 * np.sum(determinant_array)

    return area

def pol_perimeter_faster(vert_positions):
    
    # max_num_sides = tissue.face_df['num_sides'].max()
    # offset =  max_num_sides - tissue.face_df['num_sides']
    # vert_positions_df = pd.Series(vert_positions,
    #                         index=tissue.face_df.index,dtype=object)
   
    # reduced_offset = offset[offset>0]
    # indices_to_pad = reduced_offset.index

    # vert_positions_df.loc[indices_to_pad] = \
    #                     vert_positions_df.loc[indices_to_pad].combine(
    #                     reduced_offset, lambda x, y: pad_array_2d(x,y))
       
    # padded_vert_positions = np.array([*vert_positions_df.values],dtype=float)
                     
    # pos_x,pos_y = padded_vert_positions[:,:,0], padded_vert_positions[:,:,1]
    pos_x, pos_y = vert_positions[:,:,0], vert_positions[:,:,1]
    
    pos_x_roll = np.roll(pos_x,-1,axis=1)
    pos_y_roll = np.roll(pos_y,-1,axis=1)
    
    delta_x = pos_x - pos_x_roll
    delta_y = pos_y - pos_y_roll
    
    perimeter = delta_x**2 + delta_y**2
                          
    perimeter = np.sum(np.sqrt(perimeter),axis=1)
    
    return perimeter

def pol_area_faster(vert_positions):
    
    pos_x, pos_y = vert_positions[:,:,0], vert_positions[:,:,1]
    
    determinant_array = pos_x * np.roll(pos_y,-1,axis=1) - \
                        pos_y * np.roll(pos_x,-1,axis=1)
                        
    area = 0.5 * np.sum(determinant_array,axis=1)
    
    return area

# def pol_area_faster(vert_positions):
    
#     vert_positions_df = pd.Series(vert_positions,dtype=object)
    
#     series_x = vert_positions_df.apply(lambda x: x[:,0])
#     series_y = vert_positions_df.apply(lambda y: y[:,1])
    
#     rolled_series_x = vert_positions_df.apply(lambda x: np.roll(x[:,0],-1,axis=0))
#     rolled_series_y = vert_positions_df.apply(lambda y: np.roll(y[:,1],-1,axis=0))
    
#     df_area = series_x*rolled_series_y - series_y*rolled_series_x
#     area = 0.5*df_area.apply(lambda x: np.sum(x)).values

#     return area

def pol_centroid(vert_positions, return_area=False):
    
    """ Calculates centroid of a polygon.

        Definition:
        -----------
        C_x = (1/6A)\sum_i (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
        C_y = (1/6A)\sum_i (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)

        where A, x_i, and y_i are respectively the polygon's area and
        the ordered x and y coordinates from the different polygon's corners. 
        
        Parameters:
        -----------
        vert_positions : Numpy.ndarray, shape: (N,2)
            Ordered vertex position of the polygon.

        Returns:
        -----------
        centroid: Numpy.ndarray, shape: (1,2)
        area : float
            Signed polygon area.
    """

    x, y = vert_positions[:,0], vert_positions[:,1]

    determinant_array = x * np.roll(y,-1) - np.roll(x,-1) * y
    area = 0.5 * np.sum(determinant_array)
    
    area += 1e-6

    c_x = (1/(6*area)) * np.sum((x + np.roll(x,-1)) * determinant_array)
    c_y = (1/(6*area)) * np.sum((y + np.roll(y,-1)) * determinant_array)  

    if return_area:
        centroid = np.array([c_x, c_y, area])
    else:
        centroid = np.array([c_x, c_y])

    return centroid

def pol_centroid_faster(vert_positions):
    
    pos_x, pos_y = vert_positions[:,:,0], vert_positions[:,:,1]
    
    determinant_array = pos_x * np.roll(pos_y,-1,axis=1) - \
                        pos_y * np.roll(pos_x,-1,axis=1)
                        
    area = 0.5 * np.sum(determinant_array,axis=1)
    
    area += 1e-6
    
    c_x = (1/(6*area)) * np.sum(
        (pos_x + np.roll(pos_x,-1,axis=1))*determinant_array,axis=1)
    
    c_y = (1/(6*area)) * np.sum(
        (pos_y + np.roll(pos_y,-1,axis=1))*determinant_array,axis=1)
    
    # if return_area:
    #     centroid = np.dstack((c_x,c_y,area))[0]
    # else:
    #     centroid = np.dstack((c_x,c_y))[0]    
    centroid = np.dstack((c_x,c_y))[0]    
    
    return centroid

def triangle_centroid_faster(vert_positions):
    
    pos_x, pos_y = vert_positions[:,:,0], vert_positions[:,:,1]
    
    c_x = np.mean(pos_x,axis=1)
    c_y = np.mean(pos_y,axis=1)
    
    centroid = np.dstack((c_x,c_y))[0]  
    
    return centroid

def rotation_2d(points,origin=np.array([[0, 0]]),angle=0):
    """ Rotates array of 2D points around some arbitrary origin
        by any specified angle.

        TODO: Finish documentation.
        
    Returns:
        [type]: [description]
    """
    cos_phi, sin_phi = np.cos(angle), np.sin(angle)
    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi,  cos_phi]])
    return (R @ (points - origin).T + origin.T).T

def positive_quadrant_angle(pos):
    ang = math.atan2(pos[1], pos[0])
    if ang < 0:
        ang += 2*np.pi

    return ang

def vertex_to_centroid(tissue,vert_id,face_id,
                       scale_factor=1,return_distance=False):
      
    vert_position = tissue.vert_df.loc[vert_id,['x','y']].values 
    face_centroid = tissue.face_df.loc[face_id,['x','y']].values
    

    unit_vector_to_centroid = face_centroid - vert_position
    vert_to_centroid_distance = np.linalg.norm(unit_vector_to_centroid)
    unit_vector_to_centroid /= vert_to_centroid_distance + 1e-8
    
    displacement_distance = scale_factor * vert_to_centroid_distance
  
    vert_position += displacement_distance * unit_vector_to_centroid

    if return_distance:
        return vert_position, displacement_distance
    else:       
        return vert_position
    
def displace_vertex(tissue,vert_id,vector_shift=np.array([0, 0]),
                    return_distance=False):
      
    vert_position = tissue.vert_df.loc[vert_id,['x','y']].values
       
    vert_position += vector_shift
    
    displacement_distance = np.linalg.norm(vector_shift)

    if return_distance:
        return vert_position, displacement_distance
    else:       
        return vert_position
    
def line_axis_intersection_point(points,origin=np.array([[0, 0]]),
                                            axis_angle=0):
    
    translated_points = points - origin
    axis_slope = math.tan(axis_angle)
    
    side_vectors = points[:,1]-points[:,0] + 1e-6
    
    slope = side_vectors[:,1]/side_vectors[:,0]
    
    
    # y_intercepts = points[:,1][:,1] - slope * points[:,1][:,0]
    y_intercepts = translated_points[:,1][:,1] - \
                                    slope * translated_points[:,1][:,0]
    
    ratio = y_intercepts/(axis_slope - slope + 1e-8)
    
    intersection_points_x = ratio
    intersection_points_y = axis_slope*ratio
    
    intersection_points = \
                    np.dstack((intersection_points_x,intersection_points_y))
    
    intersection_points += origin
    
    intersection_points = intersection_points[0]
    
    return intersection_points

def dbond_angle(tissue,dbond_id,nematic_angle=False):
    
    vert_IDs = tissue.edge_df.loc[dbond_id,['v_out_id','v_in_id']]
    vert_positions = tissue.vert_df.loc[vert_IDs,['x','y']].values
    
    side_vectors = vert_positions[1]-vert_positions[0] + 1e-6
    
    x, y = side_vectors 
    angle = math.atan2(y,x)   
    
    if not nematic_angle:
        if angle < 0: angle += 2.0*math.pi
    else:
        if angle < 0: angle += math.pi
    
    return angle

def dbond_axis_angle(tissue,dbond_id,axis_angle=0.0):
    dbond_ang = dbond_angle(tissue,dbond_id)
    
    angle_difference = min((2.0*math.pi-dbond_ang+axis_angle)%math.pi,
                           (2.0*math.pi+dbond_ang-axis_angle)%math.pi)

    return angle_difference


def splitTensor_angleNorm(T):
    # T = A + B
    # A = [[a,-b],
    #     [b, a]]
    # B = [[c, d],
    #     [d,-c]]
    
    a = 0.5 * (T[0, 0] + T[1, 1])
    b = 0.5 * (T[1, 0] - T[0, 1])
    c = 0.5 * (T[0, 0] - T[1, 1])
    d = 0.5 * (T[0, 1] + T[1, 0])

    theta = np.arctan2(b, a)
    AabsSq = a * a + b * b

    twophi = theta + np.arctan2(d, c)
    BabsSq = c * c + d * d

    return theta, AabsSq, twophi, BabsSq

def triangulation_Q_norm(AabsSq, BabsSq):
    
    # AabsSq = triangulation.face_df['AabsSq'].values
    # BabsSq = triangulation.face_df['BabsSq'].values
    
    Qkk = AabsSq - BabsSq
    
    gamma = 1/np.sqrt(Qkk)
    
    triangles_Q_norm = np.sqrt(BabsSq)*gamma

    triangles_Q_norm = np.arcsinh(triangles_Q_norm)
    
    return triangles_Q_norm

def calculate_triangle_state_tensor_symmetric_antisymmetric(triangulation,A_0=1.0):
    
    # Define inverse of equilateral triangle tensor.
    l = math.sqrt(4 * A_0 / math.sqrt(3))
    
    # equiTriangle = [[l, 0.5 * l],
    #                 [0, 0.5 * np.sqrt(3) * l]]
    equiTriangle = [[l, -0.5 * l],
                    [0, 0.5 * np.sqrt(3) * l]]
    equiTriangleInv = np.linalg.inv(equiTriangle)
    
    E21_xy = triangulation.face_df[['dr_21_x','dr_21_y']].values
    E32_xy = triangulation.face_df[['dr_32_x','dr_32_y']].values
    
    # Convert triangle vectors into triangle state tensor R.  
    triangle_state_tensors_R = np.dstack((E21_xy, E32_xy))
    
    # Convert triangle tensor R in triangle tensor S = R.C^{-1}
    triangle_state_tensors_S = triangle_state_tensors_R @ equiTriangleInv

    # Split triangle tensor S in traceless symmetric part and trace anti-symmetric part: theta, AabsSq, twophi, BabsSq.
    triangle_state = np.array([splitTensor_angleNorm(triang_S)
                                for triang_S in triangle_state_tensors_S])
    

    return triangle_state

def fit_ellipse_to_tissue_boundary(tissue):
    boundary_vertices = tissue.vert_df.loc[tissue.vert_df['is_interior'] == False,
                                       ['x','y']].values

    pos_x = boundary_vertices[:,0]
    pos_y = boundary_vertices[:,1]

    pos_x = np.reshape(pos_x,(len(pos_x),1))
    pos_y = np.reshape(pos_y,(len(pos_y),1))

    # Formulate and solve the least squares problem ||Ax - b ||^2
    b_vector = np.ones_like(pos_x)
    # A_matrix = np.hstack([pos_x**2,2*pos_x * pos_y, pos_y**2, pos_x, pos_y])
    A_matrix = np.hstack([pos_x**2,2*pos_x * pos_y, pos_y**2, 2*pos_x, 2*pos_y])
    
    # b_vector = np.zeros_like(pos_x)
    # c_vector = np.ones_like(pos_x)
    # A_matrix = np.hstack([pos_x**2,2*pos_x * pos_y, pos_y**2, pos_x, pos_y, c_vector])
    
    
    ellipse_coefficients = np.linalg.lstsq(A_matrix, b_vector,rcond=None)[0].squeeze()
    
    return ellipse_coefficients

def find_ellipse_data(tissue):
    
    ellip_fit = fit_ellipse_to_tissue_boundary(tissue)
    
    # ellipse_center = \
    #     np.array([ellip_fit[2]*ellip_fit[3]-ellip_fit[1]*ellip_fit[4],
    #               ellip_fit[0]*ellip_fit[4]-ellip_fit[1]*ellip_fit[3]])
        
    # ellipse_center /= 2*(ellip_fit[1]**2 - ellip_fit[0]*ellip_fit[2])
    
    ellipse_center = \
        np.array([ellip_fit[2]*ellip_fit[3]-ellip_fit[1]*ellip_fit[4],
                  ellip_fit[0]*ellip_fit[4]-ellip_fit[1]*ellip_fit[3]])
        
    ellipse_center /= (ellip_fit[1]**2 - ellip_fit[0]*ellip_fit[2])
    
    ellipise_angle = 2*ellip_fit[1]/(ellip_fit[2]-ellip_fit[0])
    
    ellipise_angle = 0.5*math.atan(-ellipise_angle)
    
    # mu = 1/(ellip_fit[0]*ellipse_center[0]**2 +
    #         2*ellip_fit[1]*ellipse_center[0]*ellipse_center[1] + 
    #         ellip_fit[2]*ellipse_center[1]**2 - 1)
    
    # m11 = mu*ellip_fit[0]
    # m12 = mu*ellip_fit[1]
    # m22 = mu*ellip_fit[2]
    
    delta = math.sqrt((ellip_fit[0]-ellip_fit[2])**2 + 4*ellip_fit[1]**2) 
    
    gamma = ellip_fit[0]+ellip_fit[2]
    
    # return delta,-gamma
    
    # num = 2*(ellip_fit[0]*ellip_fit[4]**2 + ellip_fit[1]*ellip_fit[3]**2 +
    #          ellip_fit[1]**2 - 2*ellip_fit[1]*ellip_fit[3]*ellip_fit[4]-
    #          ellip_fit[0]*ellip_fit[2])
    
    num = 2*(ellip_fit[0]*ellip_fit[4]**2 + ellip_fit[1]*ellip_fit[3]**2 +
             -ellip_fit[1]**2 - 2*ellip_fit[1]*ellip_fit[3]*ellip_fit[4]+
             ellip_fit[0]*ellip_fit[2])
    
    dem_a = (ellip_fit[1]**2-ellip_fit[0]*ellip_fit[2])*(delta-gamma)
    dem_b = (ellip_fit[1]**2-ellip_fit[0]*ellip_fit[2])*(-delta-gamma)
    
    # return num,dem_a,dem_b
    
    a_major = math.sqrt(num/dem_a)
    b_minor = math.sqrt(num/dem_b)
    
    return ellipse_center,ellipise_angle,a_major,b_minor

def elongation_lines(triangulation,scale):
    
    triang_centers = triangulation.face_df[['x','y']].values
    twophi = triangulation.face_df['twophi'].values

    cos_two_phi, sin_two_phi = np.cos(0.5*twophi), np.sin(0.5*twophi)

    unit_vectors = np.dstack((cos_two_phi,sin_two_phi))[0]

    end_point_1 = triang_centers + scale*0.5*unit_vectors
    end_point_2 = triang_centers - scale*0.5*unit_vectors

    lines = list(zip(end_point_1,end_point_2))
    
    return lines

# def calculate_triangle_state_tensor_symmetric_antisymmetric(E12_xy, E13_xy, A_0=1.0):
    
#     # Define inverse of equilateral triangle tensor.
#     l = math.sqrt(4 * A_0 / math.sqrt(3))
    
#     equiTriangle = [[l, 0.5 * l],
#                     [0, 0.5 * np.sqrt(3) * l]]
#     equiTriangleInv = np.linalg.inv(equiTriangle)

#     # Convert triangle vectors into triangle state tensor R.
#     triangle_state_tensors_R = np.stack((E12_xy, E13_xy), axis=-1)
#     # Convert triangle tensor R in triangle tensor S = R.C^{-1}
#     triangle_state_tensors_S = np.array([np.dot(triangleR, equiTriangleInv) for triangleR in triangle_state_tensors_R])

#     # Split triangle tensor S in traceless symmetric part and trace anti-symmetric part: theta, AabsSq, twophi, BabsSq.
#     if len(np.shape(triangle_state_tensors_S)) > 2:
#         triangles_state_tensor_decomposed = np.array(list(map( splitTensor_angleNorm, triangle_state_tensors_S)))
#     else:
#         triangles_state_tensor_decomposed = np.array( splitTensor_angleNorm(triangle_state_tensors_S))

#     return triangles_state_tensor_decomposed