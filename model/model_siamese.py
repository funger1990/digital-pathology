import os
import sys
import datetime
import copy
import collections
import argparse
import configparser
import platform
import itertools
import shutil
import random
import numpy as np
import pandas as pd
import sklearn.metrics

import PIL.Image
import tifffile
import torch
import torch.utils.data
import torchvision

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ================================================
class PathologyDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, csv, transform, is_encoder=False):
        self.rootdir = rootdir
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.is_encoder = is_encoder

        if not self.is_encoder:
            self.df['p_fns'] = self.df['near_patches'].str.split(',')
            self.df['n_fns'] = self.df['far_patches'].str.split(',')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # anchor
        row = self.df.loc[idx, :]
        A = PIL.Image.open(self.rootdir + row['file'])
        A = A.convert('RGB')
        A = self.transform(A)

        if self.is_encoder:
            return A

        # positive, near patch
        P = PIL.Image.open(self.rootdir + random.choice(row['p_fns']))
        P = P.convert('RGB')
        P = self.transform(P)

        # negative, far patch
        N = PIL.Image.open(self.rootdir + random.choice(row['n_fns']))
        N = N.convert('RGB')
        N = self.transform(N)

        return A, P, N


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        df = dataset.df
        self.indices = df.index.tolist()

        # weight for each patch
        counter = df['wsi'].value_counts()
        weights = 1 / len(counter) / counter[df['wsi']]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, len(self.indices), replacement=True))

    def __len__(self):
        return len(self.indices)


# ==============================================
# Pathology model
class PathologyModel(object):
    def __init__(self, configfile, record_log=False):
        self.start_time = datetime.datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M")

        self.record_log = record_log

        assert os.path.exists(configfile), '[Error] Config does not exist'
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        # create log
        assert self.config.has_option('output', 'outdir'), '[Error] No outdir in config'
        self.outdir = self.config.get('output', 'outdir')
        os.makedirs(self.outdir, exist_ok=True)
        self.logfile = self.outdir + '/' + self.timestamp + '.log'

        # check python modules, gpu, parameters
        self.device, self.rootdir, self.train_csv, self.val_csv, self.params = self.check_module_parameter()

        # initialize ResNet 18 model
        self.model, self.criterion, self.optimizer, self.scheduler = self.init_model()

        # dataloader
        self.train_dataset, self.train_dataloader = self.make_dataloader(self.train_csv, 'train')
        self.val_dataset, self.val_dataloader = self.make_dataloader(self.val_csv, 'inference')

        # copy configfile
        if record_log:
            shutil.copyfile(configfile, self.outdir + '/' + self.timestamp + '.ini')

        self.best_model_wts = None

    def logging(self, txt, show=False, new=False):
        if self.record_log is True:
            with open(self.logfile, 'w' if new else 'a') as f:
                f.write('{}\n'.format(txt))

        if show:
            print(txt)

    def check_module_parameter(self):
        self.logging('[Start] ' + self.timestamp, new=True)

        self.logging('\n\n[Device and modules]')
        self.logging('{:15s}{}'.format('Platform', platform.platform()))
        self.logging('{:15s}{}'.format('Python', platform.python_version()))
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            self.logging('{:15s}{}, {:.1f}G'.format('GPU', gpu.name, int(gpu.total_memory) / 1024 ** 3))
            self.logging('{:15s}{}'.format('CUDA', torch.version.cuda))
            device = torch.device("cuda:0")
        else:
            self.logging('[Error] GPU not available')
            device = torch.device('cpu')
            sys.exit(1)

        self.logging('{:15s}{}'.format('torch', torch.__version__))
        self.logging('{:15s}{}'.format('torchvision', torchvision.__version__))
        self.logging('{:15s}{}'.format('PIL', PIL.__version__))
        self.logging('{:15s}{}'.format('sklearn', sklearn.__version__))

        rootdir = self.config.get('input', 'rootdir')
        train_csv = self.config.get('input', 'train_csv')
        val_csv = self.config.get('input', 'val_csv')
        self.logging('\n\n[Input/Output]')
        self.logging('{:15s}{}'.format('rootdir:', rootdir))
        self.logging('{:15s}{}'.format('train_csv:', train_csv))
        self.logging('{:15s}{}'.format('val_csv:', val_csv))
        self.logging('{:15s}{}'.format('logfile:', self.logfile), show=True)

        # parameters
        Params = collections.namedtuple(
            'Params',
            'pretrained pretrained_model mil mega_patch_size mega_patch_pool '
            'optimizer lr milestones num_epochs batch_size num_workers')

        params = Params(self.config.getboolean('parameter', 'pretrained'),
                        self.config.get('input', 'pretrained_model'),
                        self.config.getboolean('parameter', 'mil'),
                        self.config.getint('parameter', 'mega_patch_size'),
                        self.config.get('parameter', 'mega_patch_pool'),
                        self.config.get('parameter', 'optimizer'),
                        self.config.getfloat('parameter', 'lr'),
                        [int(i) for i in self.config.get('parameter', 'milestones').split(',')],
                        self.config.getint('parameter', 'num_epochs'),
                        self.config.getint('parameter', 'batch_size'),
                        self.config.getint('parameter', 'num_workers'))

        self.logging('\n\n[Parameters]')

        if params.pretrained:
            self.logging('{:15s}{}'.format('pretrained:', params.pretrained_model))
        else:
            self.logging('{:15s}{}'.format('pretrained:', params.pretrained))

        self.logging('{:15s}{}'.format('optimizer:', params.optimizer))
        self.logging('{:15s}{}'.format('lr:', params.lr))
        self.logging('{:15s}{}'.format('milestones:', params.milestones))
        self.logging('{:15s}{}'.format('num_epochs:', params.num_epochs))
        self.logging('{:15s}{}'.format('batch_size:', params.batch_size))
        self.logging('{:15s}{}'.format('num_workers:', params.num_workers))

        return device, rootdir, train_csv, val_csv, params

    def make_dataloader(self, csv, mode):
        """dataloader"""
        if mode == 'train':
            dataset = PathologyDataset(
                rootdir=self.rootdir,
                csv=csv,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(20, fill=240),
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor()
                ]))

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=ImbalancedDatasetSampler(dataset),
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers)

        elif mode == 'inference':
            dataset = PathologyDataset(
                rootdir=self.rootdir,
                csv=csv,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor()]))

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers)

        elif mode == 'encoder':
            dataset = PathologyDataset(
                rootdir=self.rootdir,
                csv=csv,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor()]),
                is_encoder=True)

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers)

        else:
            self.logging('[Error] select mode of train or inference')
            sys.exit()

        return dataset, dataloader

    def init_model(self):
        self.logging('\n\n[Training]')

        # prevent downloading parameters every time
        model = torchvision.models.resnet18(pretrained=False)
        if self.params.pretrained:
            model.load_state_dict(torch.load(self.params.pretrained_model))

        # transfer learning
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model.fc.in_features
        # logit
        # TODO: 64, 16
        model.fc = torch.nn.Linear(num_ftrs, 16)
        model = model.to(self.device)

        criterion = torch.nn.TripletMarginLoss()

        if self.params.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.params.lr,
                momentum=0.9)
        elif self.params.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.params.lr)
        else:
            self.logging('[Error] invalid optimizer')
            sys.exit(1)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.params.milestones,
            gamma=0.1)

        return model, criterion, optimizer, scheduler

    def train(self):
        self.model.train()  # Set model to training mode

        # record the best model
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch, best_loss = 0, float('inf')

        for epoch in range(self.params.num_epochs):
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.logging('{}\nEpoch {:2}/{}     lr: {}'.format(
                '-' * 30, epoch + 1, self.params.num_epochs, lr), show=True)

            num_samples = 0
            running_loss = 0.0

            # Iterate over data.
            for A, P, N in self.train_dataloader:
                A = A.to(self.device)
                P = P.to(self.device)
                N = N.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    A_feature = self.model(A)
                    P_feature = self.model(P)
                    N_feature = self.model(N)
                    loss = self.criterion(A_feature, P_feature, N_feature)
                    loss.backward()
                    self.optimizer.step()

                num_samples += A.size(0)
                running_loss += loss.item() * num_samples

            self.scheduler.step()

            patch_loss = running_loss / num_samples
            time_elapsed = (datetime.datetime.now() - self.start_time).seconds

            self.logging('{:5s} Time:{:4.0f}m {:2.0f}s patch-loss: {:.4f}'.format(
                'Train', time_elapsed // 60, time_elapsed % 60, patch_loss), show=True)

            # infer validation set
            patch_loss = self.inference(self.val_dataloader)
            time_elapsed = (datetime.datetime.now() - self.start_time).seconds
            self.logging('{:5s} Time:{:4.0f}m {:2.0f}s patch-loss: {:.4f}'.format(
                'Val', time_elapsed // 60, time_elapsed % 60, patch_loss), show=True)

            if patch_loss < best_loss:
                best_loss = patch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(self.model.state_dict())

            # TODO
            if (epoch + 1) % 10 == 0:
                model_path = '{}/{}_epoch{}.pth'.format(self.outdir, self.timestamp, epoch + 1)
                torch.save(self.model.state_dict(), model_path)

        # best model
        self.best_model_wts = best_model_wts
        time_elapsed = (datetime.datetime.now() - self.start_time).seconds

        self.logging('=' * 30, show=True)
        self.logging('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), show=True)
        self.logging('Best Val epoch={} patch-loss={:.4f}'.format(
            best_epoch, best_loss), show=True)
        self.logging('\n\n[End] ' + datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    def inference(self, dataloader):
        self.model.eval()  # Set model to evaluate mode

        num_samples = 0
        running_loss = 0.0

        # Iterate over data.
        for A, P, N in dataloader:
            A = A.to(self.device)
            P = P.to(self.device)
            N = N.to(self.device)

            # forward
            with torch.set_grad_enabled(False):
                A_feature = self.model(A)
                P_feature = self.model(P)
                N_feature = self.model(N)
                loss = self.criterion(A_feature, P_feature, N_feature)

            num_samples += A.size(0)
            running_loss += loss.item() * num_samples

        patch_loss = running_loss / num_samples

        return patch_loss

    def encoder(self, dataloader):
        self.model.eval()

        feature_list = []
        for X in dataloader:
            X = X.to(self.device)
            # forward
            with torch.set_grad_enabled(False):
                feature = self.model(X)
            feature_list.append(feature)

        features = torch.cat(feature_list)

        return features

    def save_model(self, save):
        if save == 'best':
            self.model.load_state_dict(self.best_model_wts)
            model_path = self.outdir + '/' + self.timestamp + '_best.pth'
            torch.save(self.model.state_dict(), model_path)
        elif save == 'last':
            model_path = self.outdir + '/' + self.timestamp + '_last.pth'
            torch.save(self.model.state_dict(), model_path)


# ====================================================================
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c')
    parser.add_argument('--save', '-s')
    args = parser.parse_args()

    if args.config is not None:
        configfile = args.config
    else:
        configfile = 'E:/PAIP2020/code/config.ini'

    print(configfile)

    pmodel = PathologyModel(configfile, record_log=True)
    pmodel.train()

    if args.save is not None:
        pmodel.save_model(args.save)
    else:
        pmodel.save_model('best')

