import argparse
import trimesh
import numpy as np
from fire import Fire


def clip(input_file, z_threshold, output_file):
    mesh = trimesh.load(input_file)
    vertex_indices_to_keep = mesh.vertices[:, 2] >= z_threshold

    # Mask of faces that should be kept. All the vertices of the face should satisfy the condition.
    faces_mask = np.all(vertex_indices_to_keep[mesh.faces], axis=1)
    new_faces = mesh.faces[faces_mask]

    # New vertices array after removing the unwanted vertices
    new_vertices = mesh.vertices[vertex_indices_to_keep]

    # Mapping from old vertex indices to new ones
    old_vertex_indices = np.arange(len(mesh.vertices))[vertex_indices_to_keep]
    mapping = dict(zip(old_vertex_indices, np.arange(len(new_vertices))))

    # Use the mapping to update the indices in the faces
    for i in range(len(new_faces)):
        new_faces[i] = [mapping[v] for v in new_faces[i]]

    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    new_mesh.export(output_file)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Filter a PLY mesh by Z value.")
    # parser.add_argument("--input", "-i", help="Path to the input .ply file")
    # parser.add_argument("--threshold", "-t", type=float, help="Z-value threshold")
    # parser.add_argument("--output", "-o", help="Path to the output .ply file")
    # args = parser.parse_args()

    # filter_mesh(args.input, args.threshold, args.output)
    Fire()
