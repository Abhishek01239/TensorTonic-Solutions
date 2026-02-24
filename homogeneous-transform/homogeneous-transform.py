import numpy as np

def apply_homogeneous_transform(T, points):

    # Convert to NumPy arrays
    T = np.asarray(T)
    points = np.asarray(points)

    single_input = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single_input = True

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # (N,4)

    # Apply transformation
    transformed_h = points_h @ T.T  # (N,4)

    # Drop last coordinate
    transformed = transformed_h[:, :3]

    if single_input:
        return transformed[0]

    return transformed