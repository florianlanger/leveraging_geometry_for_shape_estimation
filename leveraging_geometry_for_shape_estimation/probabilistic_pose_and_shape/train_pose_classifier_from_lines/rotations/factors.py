import torch
from probabilistic_formulation.utilities import create_all_possible_combinations_3,create_all_possible_combinations_2,create_all_possible_combinations_4,create_all_possible_combinations_2_dimension_1
import numpy as np

def factor_reproject_lines_angle_no_division_z(line_dirs_3D,lines_2D,R,B):
    
    """
    lines_2D: N x 4 (x1,y1,x2,y2)
    line_dirs_3D: N x 3 (nx,ny,nz) 
    R: Nx3x3, 
    B: HxWx3 (pixel bearing real image)
    f: focal length"""
    # print('lines 2d',lines_2D)
    assert (lines_2D >= 0).all()

    N = line_dirs_3D.shape[0]

    assert len(line_dirs_3D.shape) == 2 and line_dirs_3D.shape[0] == N and line_dirs_3D.shape[1] == 3
    assert len(lines_2D.shape) == 2 and lines_2D.shape[0] == N and lines_2D.shape[1] == 4
    assert len(R.shape) == 3 and R.shape[0] == N and R.shape[1] == 3 and R.shape[2] == 3

    line_dirs_3D = line_dirs_3D.unsqueeze(1)


    n_p = torch.transpose(torch.matmul(R,torch.transpose(line_dirs_3D,-1,-2)),-1,-2).squeeze()

    nx,ny,nz = n_p[:,0],n_p[:,1],n_p[:,2]

    b1 = B[lines_2D[:,0],lines_2D[:,1],:]
    b2 = B[lines_2D[:,2],lines_2D[:,3],:]

    b1 = b1 / b1[:,2:3]
    b2 = b2 / b2[:,2:3]


    x1,y1 = b1[:,0],b1[:,1]
    x2,y2 = b2[:,0],b2[:,1]

    cos_theta =  ((ny-y1*nz)*(y1-y2) + (x1*nz-nx)*(x2-x1))/ (torch.sqrt((y1 - y2)**2 + (x2 - x1)**2) * torch.sqrt((ny-y1*nz)**2 + (x1*nz-nx)**2))

    clipped_cos_theta = torch.clip(cos_theta,min=-0.9999999,max=0.9999999)

    theta = torch.arccos(clipped_cos_theta)
    min_theta = torch.min(theta,torch.abs(np.pi - theta))

    # factor = torch.exp(-torch.abs(torch.arccos(cos_theta)))
    factor = torch.exp(-min_theta)
    assert not torch.isnan(factor).any(), (x1,y1,x2,y2,cos_theta,theta,torch.sqrt((y1 - y2)**2 + (x2 - x1)**2),torch.sqrt((ny-y1*nz)**2 + (x1*nz-nx)**2),b1,b2)
    # print('factor',factor)
    return factor#,x1,y1,x2,y2,n_p,cos_theta

def get_factor_reproject_lines_multiple_R_v2(R,line_dirs_3D,lines_2D,B,mask_elements=False):

    assert len(line_dirs_3D.shape) == 2 and line_dirs_3D.shape[1] == 3
    assert len(lines_2D.shape) == 2 and lines_2D.shape[1] == 4
    assert len(R.shape) == 3 and R.shape[1] == 3 and R.shape[2] == 3

    n_3d = line_dirs_3D.shape[0]
    n_2d = lines_2D.shape[0]
    n_R = R.shape[0]

    # max_n = 40000
    max_n = 400000

    r_intervals = int(np.ceil(n_3d * n_2d * n_R /max_n))
    n_r_batch = int(np.floor(n_R / r_intervals))



    factor_all_Rs = torch.zeros_like(R)[:,0,0]
    factor_all_Rs_all_2d_lines = factor_all_Rs.unsqueeze(1).repeat(1,n_2d)
    all_masks = factor_all_Rs_all_2d_lines.to(bool)


    for i in range(r_intervals):
        R_part = R[i*n_r_batch:(i+1)*n_r_batch,:,:]
        r_part_size = R_part.shape[0]
        R_batch,lines_2D_batch,line_dirs_3D_batch = create_all_possible_combinations_3(R_part,lines_2D,line_dirs_3D)
        
        factors_all_3d_and_2d_lines = factor_reproject_lines_angle_no_division_z(line_dirs_3D_batch,lines_2D_batch,R_batch,B)
        reshaped_factors = factors_all_3d_and_2d_lines.reshape(r_part_size,n_2d,n_3d)
        factor_all_2d_lines,indices_which_direction = torch.max(reshaped_factors,dim=2)

        if mask_elements == False:
            mask = torch.ones_like(indices_which_direction).to(bool)
            # mask = factor_all_2d_lines > 0.9
        else:
            mask = torch.bitwise_and(indices_which_direction == 1,factor_all_2d_lines > 0.95)
        factor_Rs = torch.sum(factor_all_2d_lines * mask,dim=1)

        all_masks[i*n_r_batch:(i+1)*n_r_batch,:] = mask
        factor_all_Rs_all_2d_lines[i*n_r_batch:(i+1)*n_r_batch,:] = factor_all_2d_lines
        factor_all_Rs[i*n_r_batch:(i+1)*n_r_batch] = factor_Rs
    
    return factor_all_Rs,factor_all_Rs_all_2d_lines,all_masks
