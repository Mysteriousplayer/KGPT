import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import logging
from tqdm import tqdm
from utils.data_manager_new import DataManager
from utils.My_dataset import MyDataSet
from utils.toolkit import tensor2numpy
from models.LORE import LORE
from utils.get_hard_samples import *

class eth_dta(object):

    def __init__(self, args):
        super().__init__()
        if args["net_type"] == "LORE":
            self._network = LORE(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        self.args = args
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_2 = args["init_lr_2"]
        self.init_weight_decay = args["init_weight_decay"]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.class_num = self._network.class_num
        self._device = args['device'][0]
        self._multiple_gpus = args['device']
        self.pull_constraint = args['pull_constraint']
        self.pull_constraint_2 = args['pull_constraint_2']
        self.new_dir = args['new_dir']
        self.shot = args['shot']
        self.ds = args['dataset']

    def train_phase(self):
        data_manager = DataManager(self.args)
        train_dataset_all, test_dataset = data_manager.get_dataset()
        train_dataset_all_, _ = data_manager.get_dataset()
        self.select_data = np.load(self.new_dir, allow_pickle=True).item()
        train_dataset = MyDataSet(self.select_data)
        self.train_loader_all = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            print(self._multiple_gpus)
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_all, train_dataset_all, train_dataset_all_)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_all, train_dataset_all, train_dataset_all_):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            # print(name)
            param.requires_grad_(False)
            if "classifier" in name:
                param.requires_grad_(True)
            if "global_p" in name:
                param.requires_grad_(True)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "WB" in name:
                param.requires_grad_(True)
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch, eta_min=self.args['lr_min'])
        self.run_epoch = self.init_epoch
        self.train_function(train_loader, test_loader, optimizer, scheduler, train_loader_all, train_dataset_all, train_dataset_all_)


    def train_function(self, train_loader, test_loader, optimizer, scheduler,train_loader_all, train_dataset_all, train_dataset_all_):
        ########################## easy stage ##################################
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            losses = 0.
            correct, total, k_score_train = 0, 0, 0
            losses2, losses3 = 0, 0
            for i, (inputs, targets, p_targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                p_targets = p_targets.to(self._device)

                outputs = self._network(inputs, target=targets, p_target=p_targets)
                logits = outputs['logits']
                loss = F.cross_entropy(logits, targets)
                loss2 = torch.mean(outputs['increase_sim'])
                ##################################################################
                loss3 = torch.mean(outputs['reduce_sim'])
                ##################################################################
                loss = loss - self.pull_constraint * loss2 + self.pull_constraint_2 * loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses2 += self.pull_constraint * loss2.item()
                losses3 += self.pull_constraint_2 * loss3.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            print('lr', scheduler.get_lr())
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Epoch {}/{} => Loss {:.3f}, Loss2 {:.3f}, Loss3 {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                epoch + 1, self.run_epoch, losses / len(train_loader), losses2 / len(train_loader),
                losses3 / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

        ########################## hard stage ##################################
        self._network.train()
        self.run_epoch = 5
        self.init_epoch = self.run_epoch
        prog_bar = tqdm(range(self.run_epoch))
        optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr_2,weight_decay=self.init_weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch,eta_min=self.args['lr_min'])

        for _, epoch in enumerate(prog_bar):
            if epoch == 0:
                if self.ds == 'cifar':
                    new_data = get_hard_sample_cifar(self._network, train_dataset_all, train_dataset_all_, self._device,self.shot)
                elif self.ds == 'cifar10':
                    new_data = get_hard_sample_cifar10(self._network, train_dataset_all, train_dataset_all_, self._device, self.shot)
                else:
                    new_data = get_hard_sample(self._network, train_dataset_all, train_dataset_all_, self._device, self.shot)
                new_train_dataset = MyDataSet(new_data)
                train_loader = DataLoader(new_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            losses = 0.
            correct, total = 0, 0
            losses2, losses3 = 0, 0
            for i, (inputs, targets, p_targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                p_targets = p_targets.to(self._device)
                outputs = self._network(inputs, target=targets, p_target=p_targets)
                logits = outputs['logits']
                loss = F.cross_entropy(logits, targets)
                loss2 = torch.mean(outputs['increase_sim'])
                loss3 = torch.mean(outputs['reduce_sim'])
                ##################################################################
                loss = loss - self.pull_constraint * loss2 + self.pull_constraint_2 * loss3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses2 += self.pull_constraint * loss2.item()
                losses3 += self.pull_constraint_2 * loss3.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            print('lr', scheduler.get_lr())
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Epoch {}/{} => Loss {:.3f}, Loss2 {:.3f}, Loss3 {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                epoch + 1, self.run_epoch, losses / len(train_loader), losses2 / len(train_loader),
                losses3 / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total, key_score = 0, 0, 0

        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                # outputs = model(inputs, target=None, p_target=None)
                outputs = model.module.inference(inputs, target=None, p_target=None)
                logits = outputs['logits']
            preds = torch.max(logits, dim=1)[1]
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


