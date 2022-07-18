
from asyncio import constants
import torch

def get_combined_loss(outputs,targets,config,just_classifier=False):

    constants_multiplier = config['loss']['constants_multiplier']


    labels = targets[:,11:]
    r_classification_loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs[:,11:], labels)
    r_classification_loss = r_classification_loss.unsqueeze(1)

    labels = torch.argmax(labels,dim=1)
    predicted_class = torch.argmax(outputs[:,11:],dim=1)
    probabilities = torch.softmax(outputs[:,11:],dim=1)
    r_indices = torch.argmax(probabilities,dim=1)

    correct = (labels == predicted_class)

    metrics = {'correct': correct, 'probabilities': r_indices, 'labels': labels}

    t_offset = torch.sum((targets[:,1:4] - outputs[:,1:4])**2,dim=1).unsqueeze(1)
    s_offset = torch.sum((targets[:,4:7] - outputs[:,4:7])**2,dim=1).unsqueeze(1)
    r_offset = torch.sum((targets[:,7:11] - outputs[:,7:11])**2,dim=1).unsqueeze(1)


    weighted_t_loss = t_offset * constants_multiplier['t']
    weighted_s_loss = s_offset * constants_multiplier['s']
    weighted_r_loss = r_offset * constants_multiplier['r']
    weighted_r_classification_loss = r_classification_loss * constants_multiplier['r_classification']

    loss = weighted_t_loss + weighted_s_loss + weighted_r_loss + weighted_r_classification_loss

    metrics['t_distance'] = t_offset ** 0.5
    metrics['s_distance'] = s_offset ** 0.5
    metrics['r_distance'] = r_offset ** 0.5
    metrics['t_correct'] = (t_offset ** 0.5 < 0.2).squeeze(1)
    metrics['s_correct'] = torch.all(torch.abs((outputs[:,4:7] + 1) / (targets[:,4:7] + 1) - 1) < 0.2,dim=1)
    metrics['t_pred'] = outputs[:,1:4]
    metrics['s_pred'] = outputs[:,4:7]
    metrics['r_pred'] = outputs[:,7:11]
    metrics['weighted_classification_loss'] = weighted_r_classification_loss
    metrics['weighted_t_loss'] = weighted_t_loss
    metrics['weighted_s_loss'] = weighted_s_loss
    metrics['weighted_r_loss'] = weighted_r_loss

    return loss,metrics

def get_criterion(config):
    if config['data']['targets'] == 'labels':
        criterion = torch.nn.BCELoss(reduction='none')
    elif config['data']['targets'] == 'offsets':
        criterion = torch.nn.MSELoss(reduction='none')
    return criterion