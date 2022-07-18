
import torch
from scipy.spatial.transform import Rotation as scipy_rot

def get_device(a):

    if a.is_cuda:
        gpu_n = a.get_device()
        device = torch.device("cuda:{}".format(gpu_n))
    else:
        device = torch.device("cpu")
    return device

def quaternion_to_matrix_scipy(quat):
    assert len(quat.shape) == 2
    assert quat.shape[1] == 4

    num_rotations = quat.shape[0]
    matrix = quat[:,:3].unsqueeze(2).repeat(1,1,3) * 0

    length_quat = torch.linalg.norm(quat,dim=1).unsqueeze(1).repeat(1,4)

    quat = quat / length_quat

    for ind in range(num_rotations):
            x = quat[ind, 0]
            y = quat[ind, 1]
            z = quat[ind, 2]
            w = quat[ind, 3]

            x2 = x * x
            y2 = y * y
            z2 = z * z
            w2 = w * w

            xy = x * y
            zw = z * w
            xz = x * z
            yw = y * w
            yz = y * z
            xw = x * w

            matrix[ind, 0, 0] = x2 - y2 - z2 + w2
            matrix[ind, 1, 0] = 2 * (xy + zw)
            matrix[ind, 2, 0] = 2 * (xz - yw)

            matrix[ind, 0, 1] = 2 * (xy - zw)
            matrix[ind, 1, 1] = - x2 + y2 - z2 + w2
            matrix[ind, 2, 1] = 2 * (yz + xw)

            matrix[ind, 0, 2] = 2 * (xz + yw)
            matrix[ind, 1, 2] = 2 * (yz - xw)
            matrix[ind, 2, 2] = - x2 - y2 + z2 + w2

    return matrix


def quaternion_to_matrix_scipy_batch(quat):
    assert len(quat.shape) == 2
    assert quat.shape[1] == 4

    num_rotations = quat.shape[0]
    matrix = quat[:,:3].unsqueeze(2).repeat(1,1,3) * 0

    length_quat = torch.linalg.norm(quat,dim=1).unsqueeze(1).repeat(1,4)

    quat = quat / length_quat

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix[:, 0, 0] = x2 - y2 - z2 + w2
    matrix[:, 1, 0] = 2 * (xy + zw)
    matrix[:, 2, 0] = 2 * (xz - yw)

    matrix[:, 0, 1] = 2 * (xy - zw)
    matrix[:, 1, 1] = - x2 + y2 - z2 + w2
    matrix[:, 2, 1] = 2 * (yz + xw)

    matrix[:, 0, 2] = 2 * (xz + yw)
    matrix[:, 1, 2] = 2 * (yz - xw)
    matrix[:, 2, 2] = - x2 - y2 + z2 + w2

    return matrix

def get_gt_transforms(extra_infos,device):
    target_T = torch.Tensor(extra_infos['T_gt']).to(device)
    target_S = torch.Tensor(extra_infos['S_gt']).to(device)
    target_R = torch.Tensor(extra_infos['R_gt']).to(device)
    return target_T,target_S,target_R

def get_pred_transforms(extra_infos,outputs,device):
    pred_T = torch.Tensor(extra_infos['T']).to(device) + outputs[:,1:4]
    pred_S = torch.Tensor(extra_infos['S']).to(device) + outputs[:,4:7]


    # r_offset_pred_slow = torch.zeros((outputs.shape[0],3,3))
    # for i in range(outputs.shape[0]):
    #     r_offset_pred_slow[i] = torch.Tensor(scipy_rot.from_quat(outputs[i,7:11].detach().cpu().numpy()).as_matrix())
    # r_offset_pred = quaternion_to_matrix_scipy(outputs[:,7:11])
    r_offset_pred_batch = quaternion_to_matrix_scipy_batch(outputs[:,7:11])

    # assert (torch.abs(r_offset_pred_slow.to(device) - r_offset_pred) < 0.0001).all(),(r_offset_pred_slow,r_offset_pred)
    # assert (torch.abs(r_offset_pred_batch - r_offset_pred) < 0.0001).all(),(r_offset_pred_batch,r_offset_pred)


    pred_R = torch.matmul(torch.Tensor(extra_infos['R']).to(device),r_offset_pred_batch)

    return pred_T,pred_S,pred_R

def transform_unit_cube(T,S,R,unit_cube):
    assert len(T.shape) == 2
    assert T.shape[1] == 3
    assert len(S.shape) == 2
    assert S.shape[1] == 3
    assert len(R.shape) == 3
    assert R.shape[1] == 3
    assert R.shape[2] == 3

    # unit cube shape is (N,8,3)
    T = T.unsqueeze(1).repeat(1,8,1)
    S = S.unsqueeze(1).repeat(1,8,1).unsqueeze(3)
    R = R.unsqueeze(1).repeat(1,8,1,1)
    unit_cube = S * unit_cube
    unit_cube = torch.matmul(R,unit_cube).squeeze(3) + T
    return unit_cube

def cube_loss(pred_T,pred_S,pred_R,target_T,target_S,target_R,n_batch,device=None):

    unit_cube = torch.Tensor([[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,-1],[1,1,1],[1,-1,1],[-1,1,1],[-1,-1,1]])
    unit_cube = unit_cube.unsqueeze(0).repeat(n_batch,1,1).unsqueeze(3)
    unit_cube = unit_cube.to(device) * 0.5

    transformed_gt_cube = transform_unit_cube(target_T,target_S,target_R,unit_cube)
    transformed_pred_cube = transform_unit_cube(pred_T,pred_S,pred_R,unit_cube)

    loss = torch.sum(torch.sum((transformed_gt_cube - transformed_pred_cube)**2,dim=2),dim=1)
    loss = loss.unsqueeze(1)

    return loss

def geometric_loss(outputs,targets,config,extra_infos):



    device = get_device(outputs)

    # loss transformation of unit cube
    target_T,target_S,target_R = get_gt_transforms(extra_infos,device)
    pred_T,pred_S,pred_R = get_pred_transforms(extra_infos,outputs,device)


    loss = cube_loss(pred_T,pred_S,pred_R,target_T,target_S,target_R,outputs.shape[0],device)



    # all dummy variables
    dummy_s_r_t_offset = torch.sum((outputs[:,1:4])**2,dim=1).unsqueeze(1) * 0
    probabilities = torch.sigmoid(outputs[:,0:1])
    labels = targets[:,0:1]
    binary_prediction = (probabilities > 0.5)
    binary_labels = (labels > 0.5)
    correct = binary_prediction == binary_labels

    metrics = {'correct': correct, 'probabilities': probabilities, 'labels': labels}

    metrics['t_distance'] = dummy_s_r_t_offset
    metrics['s_distance'] = dummy_s_r_t_offset
    metrics['r_distance'] = dummy_s_r_t_offset
    metrics['t_correct'] = (dummy_s_r_t_offset < 0.2).squeeze(1)
    metrics['s_correct'] = (dummy_s_r_t_offset < 0.2).squeeze(1)
    metrics['t_pred'] = outputs[:,1:4]
    metrics['s_pred'] = outputs[:,4:7]
    metrics['r_pred'] = outputs[:,7:11]
    metrics['weighted_classification_loss'] = dummy_s_r_t_offset
    metrics['weighted_t_loss'] = dummy_s_r_t_offset
    metrics['weighted_s_loss'] = dummy_s_r_t_offset
    metrics['weighted_r_loss'] = dummy_s_r_t_offset


    return loss,metrics

def get_criterion(config):
    if config['data']['targets'] == 'labels':
        criterion = torch.nn.BCELoss(reduction='none')
    elif config['data']['targets'] == 'offsets':
        criterion = torch.nn.MSELoss(reduction='none')
    return criterion

def test_cube_loss_01():
    n_batch = 20
    pred_T = torch.rand(n_batch,3)
    pred_S = torch.rand(n_batch,3)
    pred_R = torch.rand(n_batch,3,3)

    loss = cube_loss(pred_T,pred_S,pred_R,pred_T,pred_S,pred_R,n_batch)

    assert (torch.abs(loss) < 0.0001).all(), loss
    print('test_cube_loss_01 passed')

def test_cube_loss_02():
    n_batch = 20
    pred_T = torch.rand(n_batch,3)
    pred_S = torch.ones(n_batch,3)
    pred_R = torch.rand(n_batch,3,3)

    gt_T = pred_T + torch.Tensor([[0,0,2]]).repeat(n_batch,1)

    loss = cube_loss(pred_T,pred_S,pred_R,gt_T,pred_S,pred_R,n_batch)

    assert (torch.abs(loss - 32) < 0.0001).all(), loss
    print('test_cube_loss_02 passed')

def test_cube_loss_03():
    n_batch = 20
    pred_T = torch.rand(n_batch,3)
    pred_S = torch.ones(n_batch,3)
    pred_R = torch.eye(3).unsqueeze(0).repeat(n_batch,1,1)

    gt_S = torch.ones(n_batch,3) * 2
    loss = cube_loss(pred_T,pred_S,pred_R,pred_T,gt_S,pred_R,n_batch)

    assert (torch.abs(loss - 6) < 0.0001).all(), loss
    print('test_cube_loss_03 passed')

def test_cube_loss_04():
    n_batch = 20
    pred_T = torch.rand(n_batch,3)
    pred_S = torch.ones(n_batch,3)
    pred_R = torch.eye(3).unsqueeze(0).repeat(n_batch,1,1)
    gt_R = torch.Tensor([[0,-1,0],[1,0,0],[0,0,1]]).unsqueeze(0).repeat(n_batch,1,1)

    loss = cube_loss(pred_T,pred_S,pred_R,pred_T,pred_S,gt_R,n_batch)

    assert (torch.abs(loss - 8) < 0.0001).all(), loss
    print('test_cube_loss_04 passed')





if __name__ == '__main__':
    test_cube_loss_01()
    test_cube_loss_02()
    test_cube_loss_03()
    test_cube_loss_04()