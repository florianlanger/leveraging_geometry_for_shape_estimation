import json
import numpy as np
import matplotlib
from numpy.lib.npyio import save
matplotlib.use('Agg')
import torch
from torchvision.models import vgg16
import sys
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as scipy_rot
from matplotlib import pyplot as plt
import os
from datetime import datetime
import cv2
from collections import Counter

import sys

from leveraging_geometry_for_shape_estimation.retrieval.global_pooling_model import Global_Pooling_Model


class Converter(object):

    def __init__(self,path_models_file,elev,azim):
            'Initialization'

            # open all models file
            with open(path_models_file,'r') as json_file:
                models_list = json.load(json_file)

            self.names = []
            self.categories = []
            self.models = []
            self.elev = elev
            self.azim = azim

            for i in range(len(models_list["models"])):
                
                dictionary = models_list["models"][i]
                
                self.names.append(dictionary["name"])
                self.categories.append(dictionary["category"])
                self.models.append(dictionary["model"])

            self.number_orientations = len(elev) * len(azim)
            self.number_rendered_images = len(self.names) * self.number_orientations
            self.models_per_category = Counter(self.categories)
            used = set()
            self.distinct_categories = [x for x in self.categories if x not in used and (used.add(x) or True)]
            start_indices_category = [sum([self.models_per_category[category] for category in self.distinct_categories[:i]]) for i in range(0,len(self.distinct_categories)+1)]
            self.category_to_model_indices = {self.distinct_categories[i]:(start_indices_category[i],start_indices_category[i+1]) for i in range(len(self.distinct_categories))}

    def index_to_info_dict(self,index):
        orientation = index % self.number_orientations
        model_index = index // self.number_orientations

        elev_index = orientation // len(self.azim)
        azim_index = orientation % len(self.azim)
        elev_current = str(int(self.elev[elev_index])).zfill(3)
        azim_current = str(np.round(self.azim[azim_index],1)).zfill(3)

        info_dict = {
            "category": self.categories[model_index],
            "name": self.names[model_index],
            "model": self.models[model_index],
            "orientation": int(orientation),
            "azim": azim_current,
            "elev": elev_current
        }

        return info_dict
    
    def info_dict_to_index(self,info_dict):
        model_index = self.names.index(info_dict["name"])
        return int(model_index * self.number_orientations + info_dict["orientation"])

    def category_to_index_range(self,category):
        # Note returned indices are for slicing arrays, stop index will be index of first model of DIFFERENT category
        start_index = self.categories.index(category) * self.number_orientations
        stop_index = (len(self.categories) - self.categories[::-1].index(category)) * self.number_orientations
        return (start_index, stop_index) 

    
    def name_to_index_range(self,name):
        n = self.names.index(name)
        return (n * self.number_orientations, (n+1) * self.number_orientations)





def load_network(config,device,checkpoint_path):

    pretrained_model = vgg16(pretrained=False)
    network = Global_Pooling_Model(pretrained_model,config,None)
    network.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
    network.to(device)
    network.eval()
    return network

def embed_real(network,img_path,device):
    image = imageio.imread(img_path).astype(np.float32)/255.
    padded_image = np.zeros((256,256,3))     
    padded_image[53:203,53:203,:] = image
    padded_image = np.moveaxis(padded_image,[0,1,2],[1,2,0])
    # padded_image = np.moveaxis(image,[0,1,2],[1,2,0]) 
    padded_image = torch.Tensor(padded_image).unsqueeze(0).to(device)
    embed,cluster_assign = network(padded_image,'real',None)
    embedding_real = embed[0,:]

    return embedding_real.cpu()


def embed_rendered(network,converter,path_CAD_renders,device):
    n_syn = converter.number_rendered_images
    # n_syn = 1000
    embedding_syn = torch.zeros((n_syn,512))
    print("compute syn model embeddings")
    for i in tqdm(range(n_syn)):
        info_dict = converter.index_to_info_dict(i)

        image_path = '{}/{}/{}/elev_{}_azim_{}.png'.format(path_CAD_renders,info_dict["category"],info_dict["name"].replace(info_dict["category"] + '_',''),info_dict["elev"],info_dict["azim"])
        image = imageio.imread(image_path)[:,:,:3].astype(np.float32)/255.
        image = np.moveaxis(image,[0,1,2],[1,2,0])
        image = torch.Tensor(image).unsqueeze(0)
        embed,cluster_assign = network(image.to(device),'syn',None)
        embedding_syn[i] = embed[0,:]
    return embedding_syn.cpu()
    


def euclidean_dist(relevant_model_embeddings,real_image_embed):
    return torch.norm(relevant_model_embeddings - real_image_embed, dim=1, p=None)


def find_nearest_neighbours(real_embedding,syn_embeddings,category,number_nearest_neighbours,converter):
    (start_category,end_category) = converter.category_to_index_range(category)
    relevant_model_embeddings = syn_embeddings[start_category:end_category,:]
    distances = euclidean_dist(relevant_model_embeddings,real_embedding)

    number_nn = min([number_nearest_neighbours,distances.shape[0]])
    sorted_distances,nearest_neighbours = distances.topk(number_nn, largest=False)

    nn_dict = {}
    nn_dict["nearest_neighbours"] = []
    for j in range(number_nn):
        nn_info = converter.index_to_info_dict(start_category+nearest_neighbours[j].item())
        nn_info["path"] = '{}/{}/elev_{}_azim_{}.png'.format(nn_info["category"],nn_info["name"].replace(nn_info["category"] + '_',''),nn_info["elev"],nn_info["azim"])
        nn_info["model_index"] = start_category+nearest_neighbours[j].item() 
        nn_info["distance"] = float(sorted_distances[j].item())
        nn_dict["nearest_neighbours"].append(nn_info)

    return nn_dict


def filter_nearest_neighbours(nn_dict):

    filtered = []
    seen_models = []

    for i in range(len(nn_dict["nearest_neighbours"])):
        if nn_dict["nearest_neighbours"][i]["model"] not in seen_models:
            filtered.append(nn_dict["nearest_neighbours"][i])
            seen_models.append(nn_dict["nearest_neighbours"][i]["model"])


    return {"nearest_neighbours":filtered}


def visualise(nn,img_path,path_CAD_renders,save_path):

    fig = plt.figure(figsize=(6,6))
    plt.axis('off')

    n = min([len(nn["nearest_neighbours"])+1,16])


    fig.add_subplot(4,4,1)
    image = plt.imread(img_path)
    plt.imshow(image)
    plt.axis('off')

    for j in range(1,n):
        nn_info = nn["nearest_neighbours"][j-1]
        fig.add_subplot(4,4,j+1)
        image = plt.imread(path_CAD_renders + '/' + nn_info['path'])
        plt.title('{} {:.2f} {} elev{} azim{}'.format(j,nn_info["distance"],nn_info["name"],nn_info["elev"],nn_info["azim"]),fontsize=4.5)
        plt.imshow(image)
        plt.axis('off')

    fig.savefig(save_path)

    plt.close(fig)



def get_dummy_config():
    dummy_config_old_code = {}

    dummy_config_old_code["training"] = {}
    dummy_config_old_code["training"]["compute_clusters"] = "False"
    dummy_config_old_code["training"]["encoder_weights_fixed"] = "True"

    dummy_config_old_code["model"] = {}
    dummy_config_old_code["model"]["embedding_dim"] = 512
    dummy_config_old_code["model"]["number_encoders"] = 1
    dummy_config_old_code["model"]["pooling"] = "max"
    dummy_config_old_code["model"]["filter_features"] = "False"
    return dummy_config_old_code

if __name__ == "__main__":   

    print('Get nn retrieval')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["retrieval"]["gt"] == "False":

        target_folder = global_config["general"]["target_folder"]
        checkpoint_file = global_config["retrieval"]["checkpoint_file"]
        gpu = global_config["general"]["gpu"]
        number_nn = global_config["retrieval"]["number_nearest_neighbours"]

        with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
            visualisation_list = json.load(f)

        
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(gpu))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        dummy_config_old_code = get_dummy_config()

        # path_models_file = global_config["general"]["models_folder_read"] + '/models/model_list_old_order.json'
        print('CHANGED to NORMAL ORDER')
        path_models_file = global_config["general"]["models_folder_read"] + '/models/model_list.json'
        # path_models_file = global_config["general"]["models_folder_read"] + '/models/model_list_only_bed.json'

        # path_models_file = target_folder + '/models/model_list.json'
        elev = global_config["models"]["elev"]
        azim = global_config["models"]["azim"]

        path_CAD_renders = global_config["general"]["models_folder_read"] + '/models/render_black_background'
        # path_CAD_renders = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/rendered_models/model_blender_256_black_background'


        converter = Converter(path_models_file,elev,azim)

        network = load_network(dummy_config_old_code,device,checkpoint_file)
        
        with torch.no_grad():
            true_false_to_gt_predicted = {"True":"gt","False":"predicted","roca":"predicted"}
            mask_type = true_false_to_gt_predicted[global_config["segmentation"]["use_gt"]]
            syn_embedding_path = global_config["general"]["models_folder_read"] + '/models/syn_embedding_{}_{}.npy'.format(global_config["dataset"]["split"],mask_type)
            if os.path.exists(syn_embedding_path):
                print('load existing embedding! {}'.format(syn_embedding_path))
                syn_embeddings = torch.from_numpy(np.load(syn_embedding_path))
            else:
                syn_embeddings = embed_rendered(network,converter,path_CAD_renders,device)
                np.save(syn_embedding_path,syn_embeddings.numpy())


            # syn_embeddings = torch.from_numpy(np.load(global_config["general"]["models_folder_read"] + '/models/syn_embedding_old_order.npy'))
            # syn_embeddings = None
            for name in tqdm(os.listdir(target_folder + '/cropped_and_masked_small')):

                img_path = target_folder + '/cropped_and_masked_small/' + name  

                real_path = img_path.replace('/cropped_and_masked_small/','/embedding/').split('.')[0] + '.npy'

                if os.path.exists(real_path):
                    real_embedding = np.load(real_path)
                else:
                    real_embedding = embed_real(network,img_path,device)
                    np.save(real_path,real_embedding.numpy())

                # with open(target_folder + '/segmentation_infos/' + name.replace('.png','.json'),'r') as file:
                new_name =  name.split('_')[0] + '_' + name.split('_')[1] + '_' + str(int(name.split('_')[2].split('.')[0])).zfill(2) + '.json'
                # with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as file:
                with open(target_folder + '/segmentation_infos/' + new_name,'r') as file:
                    predicted_category = json.load(file)["predictions"]["category"]

                with open(target_folder + '/gt_infos/' + name.split('_')[0] + '_' + name.split('_')[1] + '.json','r') as file:
                    gt_infos = json.load(file)

                with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
                    bbox_overlap = json.load(f)
                    
                # assert predicted_category == gt_infos["objects"][bbox_overlap['index_gt_objects']]["category"]
                
                nn = find_nearest_neighbours(real_embedding,syn_embeddings,predicted_category,number_nn,converter)

                if global_config["retrieval"]["only_different_models"]:
                    nn = filter_nearest_neighbours(nn)


                if global_config["general"]["visualise"] == "True":
                    if gt_infos["img"] in visualisation_list:
                        save_path = img_path.replace('/cropped_and_masked_small/','/nn_vis/').split('.')[0] + '.png'
                        visualise(nn,img_path,path_CAD_renders,save_path)

                # with open(target_folder + '/nn_infos/' + name.replace('.png','.json'),'w') as f:
                with open(target_folder + '/nn_infos/' + new_name,'w') as f:
                    json.dump(nn, f,indent=4)



    

    



   