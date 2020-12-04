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
class PathDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, csv, transform, flip=False, rotate=0):
        self.rootdir = rootdir
        self.df = pd.read_csv(csv)
        self.patch_size = self.df['size'].iloc[0]
        self.patch_level = self.df['level'].iloc[0]
        self.transform = transform
        self.flip = flip
        self.rotate = rotate

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch_fn = self.df['file'].iloc[idx]
        X = PIL.Image.open(self.rootdir + patch_fn)
        X = X.convert('RGB')

        # flip and rotate
        if self.flip:
            X = torchvision.transforms.functional.vflip(X)
        if self.rotate != 0:
            X = torchvision.transforms.functional.rotate(
                X, self.rotate, fill=240)

        X = self.transform(X)

        # TODO: test set has no label
        y = self.df['msi'].iloc[idx]
        info = dict(self.df.iloc[idx, :])

        return X, y, info


# ==============================================
# Pathology model
class PathModel(object):
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
        self.net, self.criterion, self.optimizer, self.scheduler = self.init_model()

        # dataloader
        self.train_dataset, self.train_dataloader = self.make_dataloader(self.train_csv, 'train')
        _, self.val_dataloader = self.make_dataloader(self.val_csv, 'val')

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
            'pretrained pretrained_model optimizer lr step_size num_epochs batch_size num_workers')

        params = Params(self.config.getboolean('parameter', 'pretrained'),
                        self.config.get('input', 'pretrained_model'),
                        self.config.get('parameter', 'optimizer'),
                        self.config.getfloat('parameter', 'lr'),
                        self.config.getint('parameter', 'step_size'),
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
        self.logging('{:15s}{}'.format('step_size:', params.step_size))
        self.logging('{:15s}{}'.format('num_epochs:', params.num_epochs))
        self.logging('{:15s}{}'.format('batch_size:', params.batch_size))
        self.logging('{:15s}{}'.format('num_workers:', params.num_workers))

        return device, rootdir, train_csv, val_csv, params

    def make_dataloader(self, csv, mode, flip=False, rotate=0):
        """dataloader"""
        if mode=='train':
            dataset = PathDataset(
                rootdir=self.rootdir,
                csv=csv,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(180, fill=240),
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor()
                ]))

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=True,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers)

        elif mode == 'val':
            dataset = PathDataset(
                rootdir=self.rootdir,
                csv=csv,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor()]),
                flip=flip,
                rotate=rotate)

            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers)

        else:
            self.logging('[Error] select mode of train or val')
            sys.exit()

        return dataset, dataloader

    def init_model(self):
        self.logging('\n\n[Training]')

        # prevent downloading parameters every time
        net = torchvision.models.resnet18(pretrained=False)
        if self.params.pretrained:
            net.load_state_dict(torch.load(self.params.pretrained_model))

        # transfer learning
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = net.fc.in_features
        # one logit
        net.fc = torch.nn.Linear(num_ftrs, 1)
        net = net.to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss()

        if self.params.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.params.lr,
                momentum=0.9)
        elif self.params.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=self.params.lr)
        else:
            self.logging('[Error] invalid optimizer')
            sys.exit(1)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.params.step_size,
            gamma=0.8)

        return net, criterion, optimizer, scheduler

    def train(self):
        self.net.train()  # Set model to training mode

        # record the best model
        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_epoch, best_acc, best_loss, best_auc, best_f1 = 0, 0, 100, 0, 0

        for epoch in range(self.params.num_epochs):
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.logging('{}\nEpoch {:2}/{}     lr: {}'.format(
                '-' * 30, epoch + 1, self.params.num_epochs, lr), show=True)

            num_samples = 0
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, infos in self.train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    outputs = self.net(inputs).reshape(-1)
                    preds = (outputs > 0).to(torch.int64)
                    loss = self.criterion(outputs, labels.float())
                    loss.backward()
                    self.optimizer.step()

                num_samples += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            self.scheduler.step()

            patch_loss = running_loss / num_samples
            patch_acc = running_corrects.double() / num_samples
            time_elapsed = (datetime.datetime.now() - self.start_time).seconds

            self.logging('{:5s} Time:{:4.0f}m {:2.0f}s patch-loss: {:.4f} patch-Acc: {:.4f}'.format(
                'Train', time_elapsed // 60, time_elapsed % 60, patch_loss, patch_acc), show=True)

            # infer validation set
            patch_loss, patch_acc, wsi_auc, wsi_f1, df, df_wsi = self.infer(self.val_dataloader)
            time_elapsed = (datetime.datetime.now() - self.start_time).seconds
            self.logging('{:5s} Time:{:4.0f}m {:2.0f}s patch-loss: {:.4f} patch-Acc: {:.4f} '
                         'WSI-AUC: {:.4f} WSI-F1: {:.4f}'.format(
                'Val', time_elapsed // 60, time_elapsed % 60, patch_loss, patch_acc, wsi_auc, wsi_f1), show=True)

            if patch_acc > best_acc:
                best_epoch = epoch + 1
                best_acc = patch_acc
                best_auc = wsi_auc
                best_f1 = wsi_f1
                best_model_wts = copy.deepcopy(self.net.state_dict())

            # TODO
            # if (epoch + 1) % 25 == 0:
            #     model_path = '{}/{}_epoch{}.pth'.format(self.outdir, self.timestamp, best_epoch)
            #     torch.save(best_model_wts, model_path)

        # best model
        self.best_model_wts = best_model_wts
        time_elapsed = (datetime.datetime.now() - self.start_time).seconds

        self.logging('=' * 30, show=True)
        self.logging('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), show=True)
        self.logging('Best Val epoch={} patch-Acc={:.4f} WSI-AUC={:.4f} WSI-F1={:.4f}'.format(
            best_epoch, best_acc, best_auc, best_f1), show=True)
        self.logging('\n\n[End] ' + datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    def infer(self, dataloader):
        # TODO: test set has no label

        self.net.eval()  # Set model to evaluate mode
        res_list = []

        # Iterate over data.
        for inputs, labels, infos in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            with torch.set_grad_enabled(False):
                outputs = self.net(inputs).reshape(-1)
                loss = self.criterion(outputs, labels.float()).tolist()
                probs = torch.sigmoid(outputs).reshape(-1).tolist()

            for k in ['msi', 'x', 'y', 'size', 'level']:
                infos[k] = infos[k].tolist()

            for i in range(inputs.size(0)):
                res_list.append([
                    infos['wsi'][i], infos['msi'][i], probs[i], loss,
                    infos['x'][i], infos['y'][i], infos['size'][i], infos['level'][i]
                ])

        df = pd.DataFrame(res_list, columns=['wsi', 'msi', 'prob', 'loss', 'x', 'y', 'size', 'level'])
        df['pred'] = (df['prob'] > 0.5).astype(int)

        patch_loss = df['loss'].mean()
        patch_acc = (df['msi'] == df['pred']).mean()

        df_wsi = df.groupby(['wsi', 'msi'])['pred'].mean().to_frame()
        df_wsi.columns = ['wsi_prob']
        df_wsi = df_wsi.reset_index()
        df_wsi['wsi_pred'] = (df_wsi['wsi_prob'] > 0.5).astype(int)

        wsi_auc = sklearn.metrics.roc_auc_score(df_wsi['msi'], df_wsi['wsi_prob'])
        wsi_f1 = sklearn.metrics.f1_score(df_wsi['msi'], df_wsi['wsi_pred'])

        return patch_loss, patch_acc, wsi_auc, wsi_f1, df, df_wsi

    def infer_augment(self, csv):
        """infer original and augmented images"""
        df_list = []

        for flip, rotate in itertools.product([True, False], list(range(0, 360, 45))):
            print(flip, rotate)
            dataset, dataloader = self.make_dataloader(
                csv=csv,
                mode='val',
                flip=flip,
                rotate=rotate)

            patch_loss, patch_acc, wsi_auc, wsi_f1, df = self.infer(dataloader)
            df['flip'] = int(flip)
            df['rotate'] = rotate
            df['wsi_aug'] = df['wsi'] + '_' + df['flip'].astype(str) + '_' + df['rotate'].astype(str)
            df_list.append(df)

        df_all = pd.concat(df_list, axis=0)

        return df_all

    def save_model(self, save):
        if save == 'best':
            self.net.load_state_dict(self.best_model_wts)
            model_path = self.outdir + '/' + self.timestamp + '_best.pth'
            torch.save(self.net.state_dict(), model_path)
        elif save == 'last':
            model_path = self.outdir + '/' + self.timestamp + '_last.pth'
            torch.save(self.net.state_dict(), model_path)


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
        # configfile = 'E:/PAIP2020/model/config_2.ini'

    print(configfile)

    pmodel = PathModel(configfile, record_log=True)
    pmodel.train()

    if args.save:
        pmodel.save_model(args.save)