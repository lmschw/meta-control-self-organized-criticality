import numpy as np

import services.ServiceOrientations as ServiceOrientations


def determineMinMaxAngleOfVision(orientations, degreesOfVision):
    """
    Determines the boundaries of the field of vision of a particle.

    Params:
        - orientation (array of floats): the current orientation of the particle
        - degreesOfVision (int [0,2*np.pi]): how many degrees of its surroundings the particle can see.

    Returns:
        Two integers representing the angular boundary of vision, i.e. the minimal and maximal angle that is still visible to the particle
    """
    angularDistance = degreesOfVision / 2
    currentAngles = normaliseAngles(ServiceOrientations.computeAnglesForOrientations(orientations))
    
    minAngles = normaliseAngles(currentAngles - angularDistance)
    maxAngles = normaliseAngles(currentAngles + angularDistance)

    return minAngles, maxAngles


def isInFieldOfVision(positions, minAngles, maxAngles):
    """
    Checks if a given particle is within the field of vision of the current particle.

    Params:
        - positionParticle (array of floats): the position of the current particle in (x,y)-coordinates
        - positionCandidate (array of floats): the position of the particle that is considered as potentially visible to the current particle
        - minAngle (float): the left boundary of the field of vision
        - maxAngle (float): the right boundary of the field of vision

    Returns:
        A boolean representing whether the given particle is in the field of vision of the current particle.
    """
    posDiffs=positions-positions[:,np.newaxis,:]  
    relativeAngles = np.arctan(posDiffs[:, :, 1], posDiffs[:, :, 0])
    angles = relativeAngles % (2*np.pi)
    return ((minAngles < maxAngles) & ((angles >= minAngles) & (angles <= maxAngles))) | ((minAngles >= maxAngles) & ((angles >= minAngles) | (angles <= maxAngles)))

def normaliseAngles(angles):
    """
    Normalises the degrees of an angle to be between 0 and 2pi.

    Params:
        - angle (float): the angle in radians

    Returns:
        Float representing the normalised angle.
    """
    angles = np.where((angles < 0), ((2*np.pi)-np.absolute(angles)), angles)
    angles = np.where((angles > (2*np.pi)), (angles % (2*np.pi)), angles)

    return angles

def compute_invisibility_mask(positions, orientations, fov=2*np.pi, view_distance=np.inf, agent_radius=1, occlusion_active=False):
    return np.logical_not(compute_visibility_mask(positions, orientations, fov, view_distance, agent_radius, occlusion_active))

def compute_visibility_mask(positions, orientations, fov=2*np.pi, view_distance=np.inf, agent_radius=1, occlusion_active=False):
    angles = ServiceOrientations.computeAnglesForOrientations(orientations)
    indices = get_visible_agents(positions, angles, fov, view_distance, agent_radius, occlusion_active)
    mask = np.full((len(positions), len(positions)), False)
    for i in range(len(positions)):
        mask[i][indices[i]] = True
    np.fill_diagonal(mask, True)
    return mask

def get_visible_agents(positions, orientations, fov=2*np.pi, view_distance=np.inf, agent_radius=1, occlusion_active=False):
    """
    Determine which agents are visible (not occluded) from each agent's perspective.
    
    Args:
        positions: (n, 2) array of x, y positions
        orientations: (n,) array of orientations in radians
        animal_type: Animal
        fov: Field of view in radians (default 2*pi)
    
    Returns:
        visibility: list of lists, where visibility[i] contains indices of agents visible to agent i
    """
    n = positions.shape[0]
    visibility = []

    for i in range(n):
        pos_i = positions[i]
        orient_i = orientations[i]

        # Vector from i to all other agents
        rel_pos = positions - pos_i  # (n, 2)
        distances = np.linalg.norm(rel_pos, axis=1)
        directions = rel_pos / np.clip(distances[:, None], 1e-8, None)

        # Angle between agent i's orientation and the direction to other agents
        forward = np.array([np.cos(orient_i), np.sin(orient_i)])
        cos_angles = directions @ forward  # dot product
        angles = np.arccos(np.clip(cos_angles, -1, 1))

        # Field of view mask
        in_fov = (angles <= fov / 2) & (distances > 0) & (distances <= view_distance)

        # Filter agents in field of view
        candidates = np.where(in_fov)[0]
        candidate_distances = distances[candidates]
        sorted_indices = candidates[np.argsort(candidate_distances)]

        if occlusion_active:
            visible = []
            occluded_mask = np.zeros(n, dtype=bool)

            for j in sorted_indices:
                if not occluded_mask[j]:
                    visible.append(j)
                    # Occlude any agents behind this one, close to the same direction
                    rel_vec = positions[j] - pos_i
                    rel_dir = rel_vec / np.linalg.norm(rel_vec)
                    dot_prods = (positions - pos_i) @ rel_dir
                    proj_lens = np.abs(np.cross(positions - pos_i, rel_dir))
                    occluded = (dot_prods > distances[j]) & (proj_lens < agent_radius)
                    occluded_mask |= occluded
        else:
            visible = sorted_indices

        visibility.append(visible)

    return visibility