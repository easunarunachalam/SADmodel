from copy import deepcopy
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pyvista as pv
from tqdm.autonotebook import tqdm, trange
import xarray as xr

def all_mito_files(tiff_fn, align_img_mitograph=True):
    """
    Given the path to a tiff containing the 3d volume image of a mitochondrial network, return (A) the image data, (B) the mitochondrial surface data produced by Mitograph, and (C) the mitochondrial skeleton data produced by Mitograph.
    The mitochondrial surface and skeleton data are returned as pyvista polydata objects.
    Note that the y-coordinate of the image stack appears to be flipped relative to Mitograph segmentation results. If `align_img_mitograph` is True, flip the y-coordinate of the image to permit direct comparison of these data.
    """

    tiff_fp = Path(tiff_fn)
    img = iio.volread(tiff_fn)

    if align_img_mitograph:
        img = np.flip(img, axis=1)

    img = xr.DataArray(data=img, dims=["z", "y", "x"])

    polymesh = pv.read(tiff_fp.with_name(tiff_fp.stem + "_mitosurface.vtk"))
    skeleton = pv.read(tiff_fp.with_name(tiff_fp.stem + "_skeleton.vtk"))

    return img, polymesh, skeleton


def face_list(polymesh, all_same_nvtx=True):
    """
    Given a pyvista polydata object `polymesh` which defines a polygonal mesh, return a list of face indices and face coordinates.
    """
    n_face_entries = len(polymesh.faces)
    mesh_points = np.array(polymesh.points)
    faces_idx = []
    faces_xyz = []
    ctr = 0
    while ctr < n_face_entries:
        n_pts_curr_face = polymesh.faces[ctr]

        pts_curr_face = polymesh.faces[(ctr+1):(ctr+1+n_pts_curr_face)]
        xyz_curr_face = np.asarray([mesh_points[ipt] for ipt in pts_curr_face])

        faces_idx.append(pts_curr_face)
        faces_xyz.append(xyz_curr_face)

        ctr += (n_pts_curr_face+1)

    if all_same_nvtx:
        faces_idx = np.asarray(faces_idx)
        faces_xyz = np.asarray(faces_xyz)

    return faces_idx, faces_xyz


def skeleton_segments(skeleton, connect_skeleton_cutoff=0.25):
    """
    Given a pyvista polydata object `skeleton`, return a list of line segments (defined by two points) that define the skeleton.
    The points in the polydata object are ordered, but do not contain information about the topology of the skeleton. Proceeding sequentially down the list of points, return a line segment between two successive points if the distance between them is less than `connect_skeleton_cutoff`.
    """
    skeleton_points = np.array(skeleton.points)
    skeleton_segments = [np.array([i, j]) for (i, j) in zip(skeleton_points[:-1], skeleton_points[1:]) if np.sqrt(np.square(i-j).sum()) < connect_skeleton_cutoff]
    return skeleton_segments


def compress_mitograph_mesh(surface, skeleton, z_compress_factor, method="nearest"):
    """
    Given pyvista polydata objects `surface` and `skeleton` which define the 3d mitochondrial network surface and skeleton, respectively, return a new surface polydata object corrected for z-stretching due to anisotropy in the point spread function.

    The z-distance between each mesh point and its corresponding skeleton point is computed, then scaled down by the proportionality factor `z_compress_factor`.
    The method for determining the mesh point - skeleton point correspondence is specified by `method`.
    If method="nearest", the point in the skeleton nearest (by L2 distance) to the mesh point under consideration is considered to be the corresponding skeleton point.
    If method="direction", the dot product between the (unit) mesh point normal vector and each possible (unit) mesh point-skeleton point vector is computed, and divided by the square of the distance, and the skeleton point for which this ratio is maximized is selected.

    Note the following:
    - the skeleton is assumed to be correct and is not modified to account for stretching; that is, it is assumed that macroscopic objects are not stretched appreciably due e.g. to spherical aberration
    - faces in the polydata object are assumed to be triangular

    Surface and skeleton data may be generated with Mitograph (Rafelski et al. 2012; Viana et al. 2015; available at https://github.com/vianamp/MitoGraph)
    """

    corr_surface = deepcopy(surface)

    mesh_points = np.array(surface.points)
    faces_idx, _ = face_list(surface)

    skeleton_points = np.array(skeleton.points)
    n_skeleton_points = len(skeleton_points)

    # construct adjacency matrix for mesh points
    # assuming that each face is defined by three points
    n_mesh_points = np.max(faces_idx) + 1
    adjmx = np.zeros((n_mesh_points, n_mesh_points), dtype=bool)
    adjmx[np.arange(n_mesh_points), np.arange(n_mesh_points)] = True
    adjmx[faces_idx[:,0], faces_idx[:,1]] = True
    adjmx[faces_idx[:,1], faces_idx[:,0]] = True
    adjmx[faces_idx[:,0], faces_idx[:,2]] = True
    adjmx[faces_idx[:,2], faces_idx[:,0]] = True
    adjmx[faces_idx[:,1], faces_idx[:,2]] = True
    adjmx[faces_idx[:,2], faces_idx[:,1]] = True

    if method == "nearest":
        # for each mesh point, find nearest skeleton point
        # and store the z value of that skeleton point
        nearest_skeleton_point_z_vals = []
        for mesh_point in mesh_points:
            dxyz = np.tile(mesh_point, (n_skeleton_points,1)) - skeleton_points
            r2 = np.square(dxyz).sum(axis=1)
            r = np.sqrt(r2)
            idx_closest = r.argmin()
            nearest_skeleton_point_z_vals.append(skeleton_points[idx_closest,2])
        nearest_skeleton_point_z_vals = np.array(nearest_skeleton_point_z_vals)

    elif method == "direction":
        # for each mesh point

        corr_surface = corr_surface.compute_normals()
        point_normals = np.array(corr_surface.point_normals)
        mean_point_normals = np.zeros_like(point_normals)

        # run mean filter over point normal mesh
        for imesh_point in range(mesh_points.shape[0]):
            curr_and_neigh_point_normals = point_normals[adjmx[:,imesh_point],:]
            mean_point_normal = np.mean(curr_and_neigh_point_normals, axis=0)
            mean_point_normals[imesh_point,:] = mean_point_normal

        nearest_skeleton_point_z_vals = []
        for mesh_point, point_normal in zip(mesh_points, mean_point_normals):
            dxyz = np.tile(mesh_point, (n_skeleton_points,1)) - skeleton_points
            r2 = np.square(dxyz).sum(axis=1)
            r = np.sqrt(r2)

            normdxyz = dxyz / np.vstack([r,r,r]).T

            tile_normal = np.tile(point_normal, (n_skeleton_points,1))
            dot_normdxyz_pointnormal = np.multiply(normdxyz, tile_normal).sum(axis=1)

            # r[r > r_search_cutoff] = np.inf
            idx_closest = np.divide(dot_normdxyz_pointnormal, r**2).argmax()
            # print(idx_closest)
            nearest_skeleton_point_z_vals.append(skeleton_points[idx_closest,2])
        nearest_skeleton_point_z_vals = np.array(nearest_skeleton_point_z_vals)

    else:
        raise SyntaxError("Invalid method for skeleton point selection")

    # for each mesh point, find the median z-coordinate of it and it's nearest
    # neighbors' respective nearest skeleton points
    # use that z-coordinate to calculate a z-distance from the skeleton
    # then rescale that z-distance from the skeleton
    new_mesh_points = []
    for imesh_point in range(mesh_points.shape[0]):
        mesh_point = mesh_points[imesh_point]
        curr_and_neigh_skeleton_z_vals = nearest_skeleton_point_z_vals[adjmx[:,imesh_point]]
        median_skeleton_z_val = np.median(curr_and_neigh_skeleton_z_vals)

        dz = mesh_point[2] - median_skeleton_z_val
        new_z = median_skeleton_z_val + z_compress_factor*dz
        new_mesh_point = [mesh_point[0], mesh_point[1], new_z]
        new_mesh_points.append(new_mesh_point)
    new_mesh_points = np.array(new_mesh_points)

    corr_surface.points = new_mesh_points

    return corr_surface, faces_idx
