import os
import random
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse

parser = argparse.ArgumentParser(description='setting')
parser.add_argument('--shot', default=16, type=int, help='sample number per class')
parser.add_argument('--task', default='fs', type=str, help='fs or eth')
args = parser.parse_args()


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

# Load the model
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)
#
train_dir = os.path.join('/home/shaokun/Downloads/Aircraftimage/', 'train')
train = datasets.ImageFolder(train_dir, transform=preprocess)
print(train.classes)
test_dir = os.path.join('/home/shaokun/Downloads/Aircraftimage/', 'test') # anchor set
test = datasets.ImageFolder(test_dir, transform=preprocess)


def get_data(dataset):
    all_images = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, shuffle=False, batch_size=1000, num_workers=4)):

            all_images.append(images)
            all_labels.append(labels)
    return torch.cat(all_images), torch.cat(all_labels)

input_all, input_all_label = split_images_labels(train.imgs)

alpha_dr_herding = np.zeros((len(train.classes), 750), np.float32)

prototypes = [[] for i in range(len(train.classes))]
for orde in range(len(train.classes)):
    prototypes[orde] = input_all[np.where(input_all_label == orde)]
prototypes = np.array(prototypes, dtype=object)
print(prototypes.shape)

nb_protos_cl = args.shot
from compute_features import compute_features

x_herd = []
y_herd = []

for iter_dico in range(len(train.classes)):
    print(iter_dico)
    current_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
    test.imgs = test.samples = current_set
    evalloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False, num_workers=4)
    num_samples = len(prototypes[iter_dico])
    print(num_samples)
    mapped_prototypes = compute_features(model, evalloader, num_samples, 512, device)
    D = mapped_prototypes.T
    D = D / np.linalg.norm(D, axis=0)

    mu = np.mean(D, axis=1)
    alpha_dr_herding[iter_dico, :] = alpha_dr_herding[iter_dico, :] * 0
    w_t = mu
    iter_herding = 0
    iter_herding_eff = 0
    while not (np.sum(alpha_dr_herding[iter_dico, :] != 0) == min(num_samples, 750)) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if alpha_dr_herding[iter_dico, ind_max] == 0:
            alpha_dr_herding[iter_dico, ind_max] = 1 + iter_herding
            iter_herding += 1
        w_t = w_t + mu - D[:, ind_max]

    alph = alpha_dr_herding[iter_dico, :]
    alph = alph[:num_samples]
    alph = (alph > 0) * (alph < nb_protos_cl * 1 + 1 + 0) * ((alph % 1) == 0) * 1.
    #print(alph)

    if args.task == "fs":
        '''random'''
        # random.seed(1993)
        random_list = random.sample(range(0,num_samples),nb_protos_cl)
        random_list = np.array(random_list)
        x_herd.append(prototypes[iter_dico][random_list])
    elif args.task == "eth":
        x_herd.append(prototypes[iter_dico][np.where(alph == 1)[0]])
    y_herd.append(np.array([iter_dico] * (nb_protos_cl)))

x_herdnp = np.concatenate(x_herd, axis=0)
y_herdnp = np.concatenate(y_herd)
print(x_herdnp.shape)
print(y_herdnp.shape)


current_set = merge_images_labels(x_herdnp, y_herdnp)
test.imgs = test.samples = current_set

def get_features(dataset):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, shuffle=False, batch_size=100, num_workers=4)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)

test_features, test_labels = get_features(test)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train.classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs).to(device)
test_features /= test_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
logit_scale = model.logit_scale.exp()
similarity = (logit_scale * test_features @ text_features.T).softmax(dim=-1)

pred_prob = similarity.cpu().detach().numpy()
correct, total = 0, 0
_, preds = torch.max(similarity, dim=1)
preds = preds.cpu()
correct += preds.eq(test_labels).sum().data.numpy()


total += len(test_labels)
acc = np.around(correct * 100 / total, decimals=2)
print('zero-shot acc', acc)
print(correct)
print(total)


seq=('rgb','truelabel','predlabel')

selectdata = dict.fromkeys(seq)


input_all, input_all_label = split_images_labels(test.imgs)
print(input_all.shape)
print(input_all_label.shape)

j = 0
for i in range(len(train.classes)):
    index = np.where(test_labels == i)
    target_all = test_labels[index]
    predict_all = preds[index]
    pred_prob_v2 = pred_prob[index]

    data_use = input_all[index]
    true_label = target_all
    pred_label = predict_all

    prob_select = pred_prob_v2[np.arange(data_use.shape[0]), true_label]
    orderindex = np.argsort(-prob_select)
    numtop = nb_protos_cl
    if prob_select.shape[0] < numtop:
        print('err')
    if j == 0:
        selectdata['rgb'] = data_use[orderindex][:numtop, ]
        selectdata['truelabel'] = true_label[orderindex][:numtop, ]
        selectdata['predlabel'] = pred_label[orderindex][:numtop, ]
        j = j + 1
    else:
        selectdata['rgb'] = np.append(selectdata['rgb'], data_use[orderindex][:numtop, ], axis=0)
        selectdata['truelabel'] = np.append(selectdata['truelabel'], true_label[orderindex][:numtop, ], axis=0)
        selectdata['predlabel'] = np.append(selectdata['predlabel'], pred_label[orderindex][:numtop, ], axis=0)


print(selectdata['rgb'].shape)
print(selectdata['truelabel'].shape)
print(selectdata['predlabel'].shape)

fname = '{}_Aircraft_{}.npy'.format(args.task, args.shot)
np.save(fname, selectdata)