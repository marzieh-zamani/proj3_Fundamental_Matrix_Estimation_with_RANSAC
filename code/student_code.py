import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    # M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
    #                 [0.6750, 0.3152, 0.1136, 0.0480],
    #                 [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################

    # Matrices A and b will be constructed using the corresponding 2D and 3D points as below
    
    # Define constants and placeholders for matrices A and b
    N = int(points_2d.shape[0])
    b = np.zeros((2*N,1))
    A = np.zeros((2*N,11))

    # For every row[k] in the corresponding 2D and 3D points
    for k in range(0,N):
        row1 = 2*k
        row2 = 2*k+1
        
        # Construct row[2k] and row[2k+1] of matrices A and b
        b[row1,0] = points_2d[k,0]
        b[row2,0] = points_2d[k,1]

        A[row1,0:3] = points_3d[k,:]
        A[row1,3] = 1
        A[row1,8:11] = -points_2d[k,0]*points_3d[k,:]

        A[row2,4:7] = points_3d[k,:]
        A[row2,7] = 1
        A[row2,8:11] = -points_2d[k,1]*points_3d[k,:]

    # Solve A*M_11 = b using np.linalg.lstsq() to obtain M_11
    M11 = np.linalg.lstsq(A, b, rcond = None)[0]

    # Append M_34 = 1 to M_11 and reshape it as M_3x4
    M12 = np.append(M11, [1])
    M = M12.reshape((3, 4))

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    # cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    # Extract matrices Q and m4 from matrix M
    Q = M[0:3,0:3]
    m4 = M[:,3]
    
    # Calculate camera center (matrix cc)
    cc = -np.linalg.inv(Q) @ m4

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    # F = np.asarray([[0, 0, -0.0004],
    #                 [0, 0, 0.0032],
    #                 [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

    # Set with_normalization to True or False
    with_normalization = True

    # If with_normalization = True:  
    # Normalize points_a and points_b and obtain T_a and T_b
    if with_normalization  ==  True:
        points_a, T_a = normalization_function(points_a)
        points_b, T_b = normalization_function(points_b)
    
    # Define constants
    N = int(points_a.shape[0])

    # Extract uA, vA, uB, vB, and I_N from points_a and points_b
    uA = points_a[:,0].reshape(N,1)
    vA = points_a[:,1].reshape(N,1)
    uB = points_b[:,0].reshape(N,1)
    vB = points_b[:,1].reshape(N,1)
    I_N = np.ones((N,1))

    # Construct matrix UV using uA, vA, uB, vB, and I_N and their multiplications
    UV = np.hstack((uA*uB, vA*uB, uB, uA*vB, vA*vB, vB, uA, vA, I_N ))

    # Solve UV* F9x1 = 0 to obtain full-rank F9x1
    _, _, vh0 = np.linalg.svd(UV,full_matrices = True)
    F3 = vh0[8,:].reshape(3,3)

    # Reduce F rank from 3 to 2
    u3, s3, vh3 = np.linalg.svd(F3, full_matrices = True)
    s3[2] = 0
    s2 = np.diag(s3)
    F = np.dot(u3, np.dot( s2, vh3))

    # If with_normalization = True:  Transform F_norm to F_orig
    if with_normalization  ==  True:
        F = np.dot( np.transpose( T_b ), np.dot( F, T_a ))

    # print('F matrix:\n',F)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def normalization_function(points_a):
    """
    Calculates the following
    scale_a: Scale of points_a
    scaled_centered_a: Scaled & centered transormation of points_a
    T_a: Transformation matrix for points_a

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
 

    Returns:
    -       scale_a: Scale of points_a
    -       scaled_centered_a: Scaled & centered transormation of points_a
    -       T_a: Transformation matrix for points_a
    """

    ###########################################################################
    ###########################################################################
    
    # Define constant N
    N = points_a.shape[0]

    # Obtain center of points
    center_a = np.average(points_a,axis = 0)
    
    # Center points around the center
    centered_a = points_a-center_a.reshape(1,2)
    
    # Obtain standard deviation
    std_a = np.sqrt( (np.sum((centered_a)**2,axis = None)/(2*N)) )
    
    # Calculate scale
    scale_a = 1/std_a
    
    # Obtain scaled and centered points_a
    scaled_centered_a = scale_a*centered_a
    
    # Obtain Ts and Tc matrices
    Ts_a = np.diag(np.array([scale_a, scale_a, 1]))
    
    Tc_a = np.diag(np.array([1., 1., 1.]))
    Tc_a[0,2] = -center_a[0]
    Tc_a[1,2] = -center_a[1]

    # Calculate matrix T
    T_a = Ts_a @ Tc_a
    ###########################################################################
    ###########################################################################

    return scaled_centered_a, T_a

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best_idx fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best_idx fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    # best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    # inliers_a = matches_a[:100, :]
    # inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    # Define constants
    N = matches_a.shape[0]
    max_iter = 15000
    threshold = 0.02
    batch_size = 8

    # Define random indexes with size = (max_iter, batch_size)
    rand_idx = np.random.randint(N,size = (max_iter, batch_size))


    # Obtain UV matrix (algorithm described in previous section)
    # Extract uA, vA, uB, vB, and I_N from points_a and points_b
    uA = matches_a[:,0].reshape(N,1)
    vA = matches_a[:,1].reshape(N,1)
    uB = matches_b[:,0].reshape(N,1)
    vB = matches_b[:,1].reshape(N,1)
    I_N = np.ones((N,1))

    # Construct matrix UV using uA, vA, uB, vB, and I_N and their multiplications
    UV = np.hstack((uA*uB, vA*uB, uB, uA*vB, vA*vB, vB, uA, vA, I_N ))

    # Define placeholders 
    inlier_count = np.zeros(max_iter)

    # Iteration for k=0:max_iter
    for k in range(max_iter):

        # Estimate F matrix using random batch[k] of 8 pairs of points
        F = estimate_fundamental_matrix(matches_a[rand_idx[k,:],:],matches_b[rand_idx[k,:],:])

        # Calculate cost of estimated F using following equation
        cost_k = np.abs( UV @ F.reshape((9,1)) )

        # Obtain and save number of inliers (with cost < threshold)
        inlier_idx = cost_k < threshold
        inlier_count[k] = np.sum(inlier_idx)


    # Sort inlier_count from maximum to minimum
    sort_idx = np.argsort(-inlier_count)

    # Obtain best_batch with maximum inliers
    best_idx = sort_idx[0]
    best_batch = rand_idx[best_idx,:]

    # Estimate best_F matrix using best_batch
    best_F = estimate_fundamental_matrix(matches_a[best_batch,:], matches_b[best_batch,:])

    # Calculate best_cost of estimated best_F
    best_cost = np.abs( UV @ best_F.reshape((9,1)) )

    # Obtain and save number of inliers (with best_cost < threshold)
    inlier_idx = best_cost < threshold
    best_inlier_count = np.sum(inlier_idx)

    # Sort cost from minimum (best match) to maximum (worst match)
    index=np.argsort(best_cost[:,0])[:50]

    # Obtain best pairs of matching points with minimum cost
    inliers_a=matches_a[index,:]
    inliers_b=matches_b[index,:]

    # Print results
    print('Found', best_inlier_count,'inliers / ', N, 'points')
    print('inliers / total points :', int(100*best_inlier_count/N), '%')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b

def tune_ransac_fundamental_matrix(matches_a, matches_b):
    """
    threshold = [0.005, 0.01, 0.02, 0.04, 0.1]

    For every train_thresh_j and test_thresh_i
        Obtain and save best_inlier_count[i,j] 

    """

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    # Define constants
    N = matches_a.shape[0]
    max_iter = 15000
    threshold = [0.005, 0.01, 0.02, 0.04, 0.1]
    batch_size = 8

    # Define random indexes with size = (max_iter, batch_size)
    rand_idx = np.random.randint(N,size = (max_iter, batch_size))
    # np.save('rand_idx_.npy',rand_idx)
    # rand_idx = np.load('rand_idx_526.npy')


    # Obtain UV matrix (algorithm described in previous section)
    # Extract uA, vA, uB, vB, and I_N from points_a and points_b
    uA = matches_a[:,0].reshape(N,1)
    vA = matches_a[:,1].reshape(N,1)
    uB = matches_b[:,0].reshape(N,1)
    vB = matches_b[:,1].reshape(N,1)
    I_N = np.ones((N,1))

    # Construct matrix UV using uA, vA, uB, vB, and I_N and their multiplications
    UV = np.hstack((uA*uB, vA*uB, uB, uA*vB, vA*vB, vB, uA, vA, I_N ))

    # Define placeholders 
    inlier_count = np.zeros((len(threshold), max_iter))
    best_idx = np.zeros(len(threshold)).astype(int)
    best_inlier_count = np.zeros((len(threshold),len(threshold))).astype(int)


    # Iteration for k=0:max_iter
    for k in range(max_iter):

        # Estimate F matrix using random batch[k] of 8 pairs of points
        F = estimate_fundamental_matrix(matches_a[rand_idx[k,:],:],matches_b[rand_idx[k,:],:])

        # Calculate cost of estimated F using following equation
        cost_k = np.abs( UV @ F.reshape((9,1)) )

        
        # Obtain and save number of inliers (with cost < threshold_j)
        for j, train_thresh_j in enumerate(threshold):
            inlier_idx = cost_k < train_thresh_j
            inlier_count[j,k] = np.sum(inlier_idx)


    # For every train_thresh_j and test_thresh_i
    # Obtain and save best_inlier_count[i,j] 
    
    for j, train_thresh_j in enumerate(threshold):
        sort_idx = np.argsort(-inlier_count[j,:])
        best_idx[j] = sort_idx[0]
        best_batch = rand_idx[best_idx[j],:]
        best_F = estimate_fundamental_matrix(matches_a[best_batch,:], matches_b[best_batch,:])
        
        cost = np.abs( UV @ best_F.reshape((9,1)) )

        for i, test_thresh_i in enumerate(threshold):
            
            inlier_idx = cost < test_thresh_i
            best_inlier_count[i,j] = np.sum(inlier_idx)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_inlier_count