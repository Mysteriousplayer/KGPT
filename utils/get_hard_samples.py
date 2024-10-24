import torch
from PIL import Image
from torch.utils.data import Dataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import numpy as np
from torch.utils.data import DataLoader

def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])
    return np.array(images), np.array(labels)

# merge into trainset.imgs
def merge_images_labels(images, labels):
    images = list(images)
    labels = list(labels)
    assert (len(images) == len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    return imgs



def get_logits(model, dataset, device):
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in DataLoader(dataset, shuffle=False, batch_size=100, num_workers=4):
            # features = model.encode_image(images.to(device))
            outputs = model(images.to(device), target=labels.to(device).long(), p_target=None)
            logits = outputs['logits']
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()
    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            outputs = tg_feature_model(inputs.to(device), target=targets, p_target=None)
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(outputs['features'].cpu())
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx == num_samples)
    return features

def get_hard_sample(model, train, eval, device, shot):
    input_all, input_all_label = split_images_labels(train.imgs)
    print(input_all.shape)
    print(input_all_label.shape)

    alpha_dr_herding = np.zeros((len(train.classes), 150000), np.float32)

    prototypes = [[] for i in range(len(train.classes))]
    for orde in range(len(train.classes)):
        prototypes[orde] = input_all[np.where(input_all_label == orde)]
    prototypes = np.array(prototypes, dtype=object)
    print(prototypes.shape)

    nb_protos_cl = shot

    x_herd = []
    y_herd = []
    p_herd = []
    for iter_dico in range(len(train.classes)):
        print(iter_dico)

        current_eval_set = merge_images_labels(prototypes[iter_dico], np.array([iter_dico] * len(prototypes[iter_dico]) ))
        eval.imgs = eval.samples = current_eval_set
        evalloader = torch.utils.data.DataLoader(eval, batch_size=100, shuffle=False, num_workers=4)
        num_samples = len(prototypes[iter_dico])
        mapped_prototypes = compute_features(model, evalloader, num_samples, 512, device)
        test_logits, test_labels = get_logits(model, eval, device)
        test_preds = torch.max(test_logits, dim=1)[1]
        test_preds = test_preds.cpu()
        test_labels = test_labels.cpu()
        is_true = test_preds.eq(test_labels.expand_as(test_preds))
        is_wrong = np.array([1] * (num_samples)) - np.array(is_true)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)
        mu = np.mean(D, axis=1)
        alpha_dr_herding[iter_dico, :] = alpha_dr_herding[iter_dico, :] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(alpha_dr_herding[iter_dico, :] != 0) == min(num_samples, 150000)) and iter_herding_eff < 150000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[iter_dico, ind_max] == 0:
                alpha_dr_herding[iter_dico, ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        alph = alpha_dr_herding[iter_dico, :]
        alph = alph[:num_samples]
        alph = alph * is_wrong
        max_k = np.partition(alph, -nb_protos_cl)[-nb_protos_cl]
        alph = (alph > max_k-1) * 1.
        x_herd.append(prototypes[iter_dico][np.where(alph == 1)[0]][:nb_protos_cl])
        y_herd.append(np.array([iter_dico] * (nb_protos_cl)))
        p_herd.append(test_preds[np.where(alph == 1)[0]][:nb_protos_cl])
    x_herdnp = np.concatenate(x_herd, axis=0)
    y_herdnp = np.concatenate(y_herd)
    pred_herdnp = np.concatenate(p_herd)
    print(x_herdnp.shape)
    print(y_herdnp.shape)
    print(pred_herdnp.shape)

    seq = ('rgb','truelabel','predlabel')
    selectdata = dict.fromkeys(seq)
    selectdata['rgb'] = x_herdnp
    selectdata['truelabel'] = y_herdnp
    selectdata['predlabel'] = pred_herdnp
    return selectdata

def get_hard_sample_cifar(model, train, eval, device, shot):
    input_all = np.array(train.data)
    input_all_label = np.array(train.targets)
    print(input_all.shape)
    print(input_all_label.shape)

    alpha_dr_herding = np.zeros((100, 500), np.float32)
    prototypes = np.zeros(
        (100, 500, input_all.shape[1], input_all.shape[2], input_all.shape[3]))
    for orde in range(100):
        prototypes[orde, :, :, :, :] = input_all[np.where(input_all_label == orde)]

    print(prototypes.shape)

    nb_protos_cl = shot

    x_herd = []
    y_herd = []
    p_herd = []
    for iter_dico in range(len(train.classes)):
        print(iter_dico)
        eval.data = prototypes[iter_dico].astype('uint8')
        eval.targets = np.array([iter_dico] * len(prototypes[iter_dico]) )

        evalloader = torch.utils.data.DataLoader(eval, batch_size=100, shuffle=False, num_workers=2)
        num_samples = eval.data.shape[0]
        mapped_prototypes = compute_features(model, evalloader, num_samples, 512, device)
        test_logits, test_labels = get_logits(model, eval, device)
        test_preds = torch.max(test_logits, dim=1)[1]
        test_preds = test_preds.cpu()
        test_labels = test_labels.cpu()
        is_true = test_preds.eq(test_labels.expand_as(test_preds))
        is_wrong = np.array([1] * (num_samples)) - np.array(is_true)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)
        mu = np.mean(D, axis=1)
        alpha_dr_herding[iter_dico, :] = alpha_dr_herding[iter_dico, :] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(alpha_dr_herding[iter_dico, :] != 0) == min(num_samples, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[iter_dico, ind_max] == 0:
                alpha_dr_herding[iter_dico, ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        alph = alpha_dr_herding[iter_dico, :]
        alph = alph[:num_samples]
        alph = alph * is_wrong
        max_k = np.partition(alph, -nb_protos_cl)[-nb_protos_cl]
        alph = (alph > max_k-1) * 1.
        x_herd.append(prototypes[iter_dico, np.where(alph == 1)[0]][:nb_protos_cl])
        y_herd.append(np.array([iter_dico] * (nb_protos_cl)))
        p_herd.append(test_preds[np.where(alph == 1)[0]][:nb_protos_cl])


    x_herdnp = np.concatenate(x_herd, axis=0)
    y_herdnp = np.concatenate(y_herd)
    pred_herdnp = np.concatenate(p_herd)
    print(x_herdnp.shape)
    print(y_herdnp.shape)
    print(pred_herdnp.shape)

    seq = ('rgb','truelabel','predlabel')
    selectdata = dict.fromkeys(seq)
    selectdata['rgb'] = x_herdnp
    selectdata['truelabel'] = y_herdnp
    selectdata['predlabel'] = pred_herdnp
    return selectdata

def get_hard_sample_cifar10(model, train, eval, device,shot):
    input_all = np.array(train.data)
    input_all_label = np.array(train.targets)
    print(input_all.shape)
    print(input_all_label.shape)

    alpha_dr_herding = np.zeros((10, 5000), np.float32)
    prototypes = np.zeros(
        (10, 5000, input_all.shape[1], input_all.shape[2], input_all.shape[3]))
    for orde in range(10):
        prototypes[orde, :, :, :, :] = input_all[np.where(input_all_label == orde)]

    nb_protos_cl = shot
    x_herd = []
    y_herd = []
    p_herd = []
    for iter_dico in range(len(train.classes)):
        print(iter_dico)
        eval.data = prototypes[iter_dico].astype('uint8')
        eval.targets = np.array([iter_dico] * len(prototypes[iter_dico]) )
        evalloader = torch.utils.data.DataLoader(eval, batch_size=100, shuffle=False, num_workers=2)
        num_samples = eval.data.shape[0]
        mapped_prototypes = compute_features(model, evalloader, num_samples, 512, device)
        test_logits, test_labels = get_logits(model, eval, device)
        test_preds = torch.max(test_logits, dim=1)[1]
        test_preds = test_preds.cpu()
        test_labels = test_labels.cpu()
        is_true = test_preds.eq(test_labels.expand_as(test_preds))
        is_wrong = np.array([1] * (num_samples)) - np.array(is_true)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)
        mu = np.mean(D, axis=1)
        alpha_dr_herding[iter_dico, :] = alpha_dr_herding[iter_dico, :] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(alpha_dr_herding[iter_dico, :] != 0) == min(num_samples, 500)) and iter_herding_eff < 10000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[iter_dico, ind_max] == 0:
                alpha_dr_herding[iter_dico, ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        alph = alpha_dr_herding[iter_dico, :]
        alph = alph[:num_samples]
        alph = alph * is_wrong
        max_k = np.partition(alph, -nb_protos_cl)[-nb_protos_cl]
        alph = (alph > max_k-1) * 1.
        x_herd.append(prototypes[iter_dico, np.where(alph == 1)[0]][:nb_protos_cl])
        y_herd.append(np.array([iter_dico] * (nb_protos_cl)))
        p_herd.append(test_preds[np.where(alph == 1)[0]][:nb_protos_cl])

    x_herdnp = np.concatenate(x_herd, axis=0)
    y_herdnp = np.concatenate(y_herd)
    pred_herdnp = np.concatenate(p_herd)
    print(x_herdnp.shape)
    print(y_herdnp.shape)
    print(pred_herdnp.shape)

    seq = ('rgb','truelabel','predlabel')
    selectdata = dict.fromkeys(seq)
    selectdata['rgb'] = x_herdnp
    selectdata['truelabel'] = y_herdnp
    selectdata['predlabel'] = pred_herdnp
    return selectdata