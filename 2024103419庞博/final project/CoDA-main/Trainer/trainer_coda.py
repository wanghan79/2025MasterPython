import sys
sys.path.append("..")
import torch
import torch.nn as nn
from utils.utils import *
from utils.utils import recompute_memory, get_features
import h5py
from clustering import compute_variance, torch_kmeans


def train_model(args, model, dset_loaders, epoch):

    source_loader = dset_loaders["source_tr"]
    target_loader_unl = dset_loaders["target"]

    criterion = nn.CrossEntropyLoss().cuda()


    if model.lemniscate_s.memory_first:
        recompute_memory(0, model, model.lemniscate_s, source_loader)

    if model.lemniscate_t.memory_first:
        recompute_memory(0, model, model.lemniscate_t, target_loader_unl)

    if epoch == 0:
        
        features = torch.cat((model.lemniscate_s.memory, model.lemniscate_t.memory), 0)

        for i in range(len(args.cluster_num_list)):
            cluster_num = args.cluster_num_list[i]
            init_centroids = None

            if args.kmeans_all_features:
                cluster_labels, cluster_centroids, cluster_phi= torch_kmeans([cluster_num], features, seed=0)
                init_centroids = cluster_centroids[0]

            cluster_labels_src, cluster_centroids_src, cluster_phi_src = torch_kmeans([cluster_num],
                                                                                      model.lemniscate_s.memory,
                                                                                      init_centroids=init_centroids,
                                                                                      seed=0)
            cluster_labels_tgt, cluster_centroids_tgt, cluster_phi_tgt = torch_kmeans([cluster_num],
                                                                                      model.lemniscate_t.memory,
                                                                                      init_centroids=init_centroids,
                                                                                      seed=0)
            weight_src = cluster_centroids_src[0]
            weight_tgt = cluster_centroids_tgt[0]
            with torch.no_grad():
                model.top_layer_src[i].weight.copy_(weight_src)
                model.top_layer_tgt[i].weight.copy_(weight_tgt)

    model.train()

    for batch_idx, (inputs_src1, _, domainIDs_src, indexes_src) in enumerate(source_loader):
        try:
            inputs_tar1, _, domainIDs_tar, indexes_tar = target_loader_unl_iter.next()
        except:
            target_loader_unl_iter = iter(target_loader_unl)
            inputs_tar1, _, domainIDs_tar, indexes_tar = target_loader_unl_iter.next()

        inputs_src1,  domainIDs_src, indexes_src = inputs_src1.cuda(),  domainIDs_src.cuda(), indexes_src.cuda()
        inputs_tar1,  domainIDs_tar, indexes_tar = inputs_tar1.cuda(), domainIDs_tar.cuda(), indexes_tar.cuda()

        model.optimizer.zero_grad()
        model.optimizer_tl_src.zero_grad()
        model.optimizer_tl_tgt.zero_grad()


        logits_src1, features_src1 = model(inputs_src1, head='src')
    


        logits_tar1, features_tar1 = model(inputs_tar1, head='tgt')


        tau = 0.01
       

        loss_iss1 = 0
        loss_iss2 = 0
        loss_cca1 = 0
        loss_cca2 = 0
        for i in range(len(args.cluster_num_list)):
            if args.in_domain:
                logits_mem_src1 = model.top_layer_src[i](model.lemniscate_s.memory[indexes_src])
                loss_iss1 += criterion(logits_src1[i], (logits_mem_src1 / tau).softmax(1).detach())

                logits_mem_tar1 = model.top_layer_tgt[i](model.lemniscate_t.memory[indexes_tar])
                loss_iss2 += criterion(logits_tar1[i], (logits_mem_tar1 / tau).softmax(1).detach())
            if args.cross_domain:

                outputs_src_cls_tgt = model.top_layer_src[i](features_tar1)
                outputs_tgt_cls_tgt = model.top_layer_tgt[i](features_tar1)
                outputs_tgt_cls_src = model.top_layer_tgt[i](features_src1)
                outputs_src_cls_src = model.top_layer_src[i](features_src1)

                if args.cross_domain_softmax:
                    outputs_src_cls_tgt = outputs_src_cls_tgt.softmax(1)
                    outputs_tgt_cls_tgt = outputs_tgt_cls_tgt.softmax(1)
                    outputs_tgt_cls_src = outputs_tgt_cls_src.softmax(1)
                    outputs_src_cls_src = outputs_src_cls_src.softmax(1)
                if args.cross_domain_loss == 'l1':
                    loss_cca1 += args.lambda_cross_domain*(
                        (outputs_src_cls_tgt - outputs_tgt_cls_tgt).abs().sum(1).mean())
                    loss_cca2 += args.lambda_cross_domain * (
                        (outputs_src_cls_src - outputs_tgt_cls_src).abs().sum(1).mean())
                    
                elif args.cross_domain_loss == 'l2':
                    loss_cca1 += args.lambda_cross_domain * ((
                            (outputs_src_cls_tgt - outputs_tgt_cls_tgt) ** 2).sum(1).mean())
                    loss_cca2 += args.lambda_cross_domain * ((
                            (outputs_src_cls_src - outputs_tgt_cls_src) ** 2).sum(1).mean())
                    
                
            if (not args.in_domain) and (not args.cross_domain):
                raise InterruptedError

        loss_iss1 /= len(args.cluster_num_list)
        loss_iss2 /= len(args.cluster_num_list)
        loss_cca1 /= len(args.cluster_num_list)
        loss_cca2 /= len(args.cluster_num_list)
    

        loss_cdm = (loss_iss1 + loss_iss2) + (loss_cca1 + loss_cca2)
        loss_cdm.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        model.optimizer_tl_src.step()
        model.optimizer_tl_src.zero_grad()
        model.optimizer_tl_tgt.step()
        model.optimizer_tl_tgt.zero_grad()
        model.lemniscate_s.update_weight(features_src1.detach(), indexes_src)
        model.lemniscate_t.update_weight(features_tar1.detach(), indexes_tar)

        if batch_idx % 50 == 0:
            print(f"Step [{batch_idx}/{len(source_loader)}] loss_iss1: {loss_iss1} loss_iss2: {loss_iss2} loss_cca1: {loss_cca1} loss_cca2: {loss_cca2}")
    model.lemniscate_s.memory_first = False
    model.lemniscate_t.memory_first = False

def test_target(args, src_loader, tgt_loader, netF, path_for_save_features=None):
    netF.eval()

    print('Prepare Gallery Features.....')
    features_gallery, gt_labels_gallery = get_features(src_loader, netF)

    print('Prepare Query Features of Target Domain.....')
    features_query, gt_labels_query = get_features(tgt_loader, netF)

    if path_for_save_features:
        with h5py.File(path_for_save_features, 'w') as hf:
            hf.create_dataset('features_gallery', data=features_gallery)
            hf.create_dataset('gt_labels_gallery', data=gt_labels_gallery)
            hf.create_dataset('features_query', data=features_query)
            hf.create_dataset('gt_labels_query', data=gt_labels_query)

    map_t = cal_map_sda(features_query, gt_labels_query,
                        features_gallery, gt_labels_gallery)
    return map_t * 100
    



