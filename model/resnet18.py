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
class PathologyDataset(torch.utils.data.Dataset):
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


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset):
        self.df = dataset.df
        self.indices = list(range(len(dataset)))

        # distribution of classes in the dataset
        counter = collections.defaultdict(lambda: collections.defaultdict(int))
        for idx in self.indices:
            label = self.df['msi'].iloc[idx]
            wsi = self.df['wsi'].iloc[idx]
            counter[label][wsi] += 1

        # weight for each patch
        weights = []
        for idx in self.indices:
            label = self.df['msi'].iloc[idx]
            wsi = self.df['wsi'].iloc[idx]
            n_label = len(counter)
            n_wsi = len(counter[label])
            n_patch = counter[label][wsi]
            weights.append(1.0 / n_label / n_wsi / n_patch)

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, len(self.indices), replacement=True))

    def __len__(self):
        return len(self.indices)


class MILBatchSampler(torch.utils.data.sampler.Sampler):
    """MSI/MSS and WSI are balanced sampling
    select multiple patches from each WSI
    """
    def __init__(self, dataset, batch_size, mega_patch_size=4):
        self.batch_size = batch_size
        self.mega_patch_size = mega_patch_size

        df = dataset.df
        df['idx'] = df.index
        self.df = df
        self.patch_size = dataset.patch_size
        self.patch_level = dataset.patch_level

        # TODO: filter small WSI
        t1 = df[df['mega_patch_size'] >= self.mega_patch_size]

        t1 = t1[['msi', 'wsi']].drop_duplicates()
        t2 = t1.groupby('msi').count().reset_index()
        t2.columns = ['msi', 'n']
        wsi_df = pd.merge(t1, t2)
        wsi_df['p'] = 1.0 / 2 / wsi_df['n']

        self.wsi_df = wsi_df
        self.patch_sr = self.df.groupby('wsi')['idx'].agg(lambda x: list(x))

    def random_multi_sampler(self):
        wsi_idx_bool = np.array(np.random.multinomial(1, self.wsi_df['p']), dtype=bool)
        wsi = self.wsi_df.loc[wsi_idx_bool, 'wsi'].iloc[0]
        multi_idx = np.random.choice(self.patch_sr[wsi], self.mega_patch_size ** 2)

        return multi_idx.tolist()

    def spatial_multi_sampler(self):
        # randomly select one WSI at balanced probability
        wsi_idx_bool = np.array(np.random.multinomial(1, self.wsi_df['p']), dtype=bool)
        wsi = self.wsi_df.loc[wsi_idx_bool, 'wsi'].iloc[0]
        # all patches of the selected WSI
        df1 = self.df[self.df['wsi'] == wsi]

        # randomly select lefttop of one mega-patch
        df2 = df1[df1['mega_patch_size'] >= self.mega_patch_size]
        lefttop_idx = np.random.choice(df2['idx'], 1)
        x = int(df1.loc[lefttop_idx, 'x'])
        y = int(df1.loc[lefttop_idx, 'y'])
        step = (4 ** self.patch_level) * self.patch_size * self.mega_patch_size

        # patches of mega-patch
        df2 = df1[(df1['x'] >= x) & (df1['x'] < x + step) & (df1['y'] >= y) & (df1['y'] < y + step)]
        multi_idx = df2['idx'].tolist()

        return multi_idx

    def __iter__(self):
        batch = []
        for i in range(self.df.shape[0] // (self.mega_patch_size ** 2)):
            batch += self.spatial_multi_sampler()
            if len(batch) == self.batch_size:
                yield batch
                batch = []


class MILLoss(object):
    def __init__(self, mega_patch_size, mega_patch_pool):
        self.mega_patch_size = mega_patch_size
        self.mega_patch_pool = mega_patch_pool

    def func(self, outputs, labels):
        sigmoid_outputs = torch.sigmoid(outputs).reshape(-1, self.mega_patch_size ** 2)
        if self.mega_patch_pool == 'max':
            bag_sigmoid_outputs = sigmoid_outputs.max(dim=1).values
        elif self.mega_patch_pool == 'avg':
            bag_sigmoid_outputs = sigmoid_outputs.mean(dim=1)
        else:
            print("[Error] mega_patch_pool must be 'max' or 'avg")
            sys.exit(1)

        bag_labels = labels[::self.mega_patch_size ** 2]
        loss = torch.nn.functional.binary_cross_entropy(bag_sigmoid_outputs, bag_labels.float())

        return loss


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
        self.model, self.train_criterion, self.infer_criterion, self.optimizer, self.scheduler = self.init_model()

        # dataloader
        self.train_dataset, self.train_dataloader = self.make_dataloader(self.train_csv, 'train')
        _, self.val_dataloader = self.make_dataloader(self.val_csv, 'inference')

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

        self.logging('{:15s}{}'.format('mil:', params.mil))
        self.logging('{:15s}{}'.format('mega_patch_size:', params.mega_patch_size))
        self.logging('{:15s}{}'.format('mega_patch_pool:', params.mega_patch_pool)),
        self.logging('{:15s}{}'.format('optimizer:', params.optimizer))
        self.logging('{:15s}{}'.format('lr:', params.lr))
        self.logging('{:15s}{}'.format('milestones:', params.milestones))
        self.logging('{:15s}{}'.format('num_epochs:', params.num_epochs))
        self.logging('{:15s}{}'.format('batch_size:', params.batch_size))
        self.logging('{:15s}{}'.format('num_workers:', params.num_workers))

        return device, rootdir, train_csv, val_csv, params

    def make_dataloader(self, csv, mode, flip=False, rotate=0):
        """dataloader"""
        if mode=='train':
            dataset = PathologyDataset(
                rootdir=self.rootdir,
                csv=csv,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(180, fill=240),
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor()
                ]))

            if self.params.mil:
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_sampler=MILBatchSampler(
                        dataset, self.params.batch_size, self.params.mega_patch_size),
                    num_workers=self.params.num_workers)
            else:
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=ImbalancedDatasetSampler(dataset),
                    batch_size=self.params.batch_size,
                    num_workers=self.params.num_workers)

        elif mode=='inference':
            dataset = PathologyDataset(
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
        # one logit
        model.fc = torch.nn.Linear(num_ftrs, 1)
        model = model.to(self.device)

        if self.params.mil:
            train_criterion = MILLoss(self.params.mega_patch_size, self.params.mega_patch_pool).func
        else:
            train_criterion = torch.nn.BCEWithLogitsLoss()

        infer_criterion = torch.nn.BCEWithLogitsLoss()

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

        return model, train_criterion, infer_criterion, optimizer, scheduler

    def train(self):
        self.model.train()  # Set model to training mode

        # record the best model
        best_model_wts = copy.deepcopy(self.model.state_dict())
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
                    outputs = self.model(inputs).reshape(-1)
                    preds = (outputs > 0).to(torch.int64)
                    loss = self.train_criterion(outputs, labels.float())
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
            patch_loss, patch_acc, wsi_auc, wsi_f1, df = self.inference(self.val_dataloader)
            time_elapsed = (datetime.datetime.now() - self.start_time).seconds
            self.logging('{:5s} Time:{:4.0f}m {:2.0f}s patch-loss: {:.4f} patch-Acc: {:.4f} '
                         'WSI-AUC: {:.4f} WSI-F1: {:.4f}'.format(
                'Val', time_elapsed // 60, time_elapsed % 60, patch_loss, patch_acc, wsi_auc, wsi_f1), show=True)

            # if patch_acc > best_acc:
            #     best_acc = patch_acc
            if patch_loss < best_loss:
                best_loss = patch_loss
                best_epoch = epoch + 1
                best_auc = wsi_auc
                best_f1 = wsi_f1
                best_model_wts = copy.deepcopy(self.model.state_dict())

            # TODO
            if (epoch + 1) % 25 == 0:
                model_path = '{}/{}_epoch{}.pth'.format(self.outdir, self.timestamp, best_epoch)
                torch.save(best_model_wts, model_path)

        # best model
        self.best_model_wts = best_model_wts
        time_elapsed = (datetime.datetime.now() - self.start_time).seconds

        self.logging('=' * 30, show=True)
        self.logging('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), show=True)
        self.logging('Best Val epoch={} patch-Acc={:.4f} WSI-AUC={:.4f} WSI-F1={:.4f}'.format(
            best_epoch, best_acc, best_auc, best_f1), show=True)
        self.logging('\n\n[End] ' + datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    def inference(self, dataloader):
        # TODO: test set has no label

        self.model.eval()  # Set model to evaluate mode
        res_list = []

        # Iterate over data.
        for inputs, labels, infos in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs).reshape(-1)
                loss = self.infer_criterion(outputs, labels.float()).tolist()
                # probs = torch.softmax(outputs, 1)[:, 1].tolist()
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
        # wsi_acc = (df_wsi['msi'] == df_wsi['wsi_pred']).mean()
        wsi_f1 = sklearn.metrics.f1_score(df_wsi['msi'], df_wsi['wsi_pred'])

        df = pd.merge(df, df_wsi)

        return patch_loss, patch_acc, wsi_auc, wsi_f1, df

    def inference_augment(self, csv):
        """infer original and augmented images"""
        df_list = []

        for flip, rotate in itertools.product([True, False], list(range(0, 360, 45))):
            print(flip, rotate)
            dataset, dataloader = self.make_dataloader(
                csv=csv,
                mode='inference',
                flip=flip,
                rotate=rotate)

            patch_loss, patch_acc, wsi_auc, wsi_f1, df = self.inference(dataloader)
            df['flip'] = int(flip)
            df['rotate'] = rotate
            df['wsi_aug'] = df['wsi'] + '_' + df['flip'].astype(str) + '_' + df['rotate'].astype(str)
            df_list.append(df)

        df_all = pd.concat(df_list, axis=0)

        return df_all

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
        # configfile = 'E:/PAIP2020/model/config_2.ini'

    print(configfile)

    pmodel = PathologyModel(configfile, record_log=True)
    pmodel.train()

    if args.save:
        pmodel.save_model(args.save)
