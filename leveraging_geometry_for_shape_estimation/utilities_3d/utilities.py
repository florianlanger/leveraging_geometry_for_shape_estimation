import numpy as np

def writePlyFile(file_name, vertices, colors):

    ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
               '''
    vertices = vertices.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices, colors])
    with open(file_name, 'w') as f:
      f.write(ply_header % dict(vert_num=len(vertices)))
      np.savetxt(f, vertices, '%f %f %f %d %d %d')