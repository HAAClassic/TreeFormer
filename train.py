import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.crowd import Crowd_TC, Crowd_UL_TC

from network import pvt_cls as TCN
from losses.multi_con_loss import MultiConLoss

from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import argparse
from losses.rank_loss import RankLoss

from losses import ramps
from losses.ot_loss import OT_Loss
from losses.consistency_loss import *

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--data-dir', default='/users/k2254235/Lab/TCT/Dataset/London_103050/', help='data path')  
 
parser.add_argument('--dataset', default='TC')
parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='the weight decay')
parser.add_argument('--resume', default='', type=str, help='the path of resume training model')
parser.add_argument('--max-epoch', type=int, default=4000, help='max training epoch')
parser.add_argument('--val-epoch', type=int, default=1, help='the num of steps to log training information')
parser.add_argument('--val-start', type=int, default=0, help='the epoch start to val')
parser.add_argument('--batch-size', type=int, default=16, help='train batch size')
parser.add_argument('--batch-size-ul', type=int, default=16, help='train batch size')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--num-workers', type=int, default=0, help='the num of training process')
parser.add_argument('--crop-size', type=int, default= 256, help='the crop size of the train image')
parser.add_argument('--rl', type=float, default=1, help='entropy regularization in sinkhorn')
parser.add_argument('--reg', type=float, default=1, help='entropy regularization in sinkhorn')
parser.add_argument('--ot', type=float, default=0.1, help='entropy regularization in sinkhorn')
parser.add_argument('--tv', type=float, default=0.01, help='entropy regularization in sinkhorn')
parser.add_argument('--num-of-iter-in-ot', type=int, default=100, help='sinkhorn iterations')
parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')
parser.add_argument('--run-name', default='Treeformer_test', help='run name for wandb interface/logging')
parser.add_argument('--consistency', type=int, default=1, help='whether to norm cood when computing distance')
args = parser.parse_args()


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    gauss = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]
    gt_discretes = torch.stack(transposed_batch[3], 0)
    return images, gauss, points, gt_discretes


def train_collate_UL(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    
    return images

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_ramp)


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = (
            "SEMI/{}_12-1-input-{}_reg-{}_nIter-{}_normCood-{}".format(
                args.run_name,args.crop_size,args.reg,
                args.num_of_iter_in_ot,args.norm_cood))

        self.save_dir = os.path.join("/scratch/users/k2254235","ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str)))
            
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            self.logger.info("using {} gpus".format(self.device_count))
        else:
            raise Exception("gpu is not available")
        
        
        downsample_ratio = 4
        self.datasets = {"train": Crowd_TC(os.path.join(args.data_dir, "train_data"), args.crop_size,
                downsample_ratio, "train"), "val": Crowd_TC(os.path.join(args.data_dir, "valid_data"),
                args.crop_size, downsample_ratio, "val")}
        
        self.datasets_ul = { "train_ul": Crowd_UL_TC(os.path.join(args.data_dir, "train_data_ul"), 
                args.crop_size, downsample_ratio, "train_ul")}

                
        self.dataloaders = {
            x: DataLoader(self.datasets[x],
                collate_fn=(train_collate if x == "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(True if x == "train" else False))
            for x in ["train", "val"]}
        
        self.dataloaders_ul = {
            x: DataLoader(self.datasets_ul[x],
                collate_fn=(train_collate_UL ),
                batch_size=(args.batch_size_ul),
                shuffle=(True),
                num_workers=args.num_workers * self.device_count,
                pin_memory=(True if x == "train" else False))
            for x in ["train_ul"]}
                 

        self.model = TCN.pvt_treeformer(pretrained=False)
        
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        
        if args.resume:
            self.logger.info("loading pretrained model from " + args.resume)
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(
                    torch.load(args.resume, self.device))
        else:
            self.logger.info("random initialization")
            
        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, 
              self.device, args.num_of_iter_in_ot, args.reg)
              
        self.tvloss = nn.L1Loss(reduction="none").to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.rankloss = RankLoss().to(self.device)
        self.kl_distance = nn.KLDivLoss(reduction='none')
        self.multiconloss = MultiConLoss().to(self.device)
        
    
    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info("-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_count_consistency_l = AverageMeter()
        epoch_count_consistency_ul = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        epoch_rank_loss = AverageMeter()
        epoch_consistensy_loss = AverageMeter()
        
        self.model.train()  # Set model to training mode

        for step, (inputs, gausss, points, gt_discrete) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            gausss = gausss.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)
             
            for st, unlabel_data in enumerate(self.dataloaders_ul["train_ul"]):
                inputs_ul = unlabel_data.to(self.device)
                break
                
                
            with torch.set_grad_enabled(True):
                outputs_L, outputs_UL, outputs_normed, CLS_L, CLS_UL = self.model(inputs, inputs_ul)
                outputs_L = outputs_L[0]
                
                with torch.set_grad_enabled(False):
                    preds_UL = (outputs_UL[0][0] + outputs_UL[1][0] + outputs_UL[2][0])/3
       
                # Compute counting loss.
                count_loss = self.mae(outputs_L.sum(1).sum(1).sum(1),torch.from_numpy(gd_count).float().to(self.device))*self.args.reg
                
                # Compute OT loss.
                ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs_L, points)
                ot_loss = ot_loss* self.args.ot
                ot_obj_value = ot_obj_value* self.args.ot
                
                gd_count_tensor = (torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
                gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                tv_loss = (self.tvloss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1)* 
                    torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.tv
            
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)
                epoch_count_loss.update(count_loss.item(), N)
                epoch_tv_loss.update(tv_loss.item(), N)
                   
                # Compute ranking loss.
                rank_loss = self.rankloss(outputs_UL)*self.args.rl
                epoch_rank_loss.update(rank_loss.item(), N)
                
                # Compute multi level consistancy loss
                consistency_loss = args.consistency * self.multiconloss(outputs_UL)
                epoch_consistensy_loss.update(consistency_loss.item(), N)
                
                
                # Compute consistency count
                Con_cls_UL = (CLS_UL[0] + CLS_UL[1] + CLS_UL[2])/3
                Con_cls_L = torch.from_numpy(gd_count).float().to(self.device)
                
                count_loss_l = self.mae(torch.stack((CLS_L[0],CLS_L[1],CLS_L[2])), torch.stack((Con_cls_L, Con_cls_L, Con_cls_L)))
                count_loss_ul = self.mae(torch.stack((CLS_UL[0],CLS_UL[1],CLS_UL[2])), torch.stack((Con_cls_UL, Con_cls_UL, Con_cls_UL)))
                epoch_count_consistency_l.update(count_loss_l.item(), N)
                epoch_count_consistency_ul.update(count_loss_ul.item(), N)
                
                
                loss = count_loss + ot_loss + tv_loss + rank_loss + count_loss_l + count_loss_ul + consistency_loss
                
  
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (torch.sum(outputs_L.view(N, -1),
                              dim=1).detach().cpu().numpy())
                              
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
        
        
        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, Count Loss: {:.2f}, OT Loss: {:.2e}, TV Loss: {:.2e}, Rank Loss: {:.2f},"
                "Consistensy Loss: {:.2f},  MSE: {:.2f}, MAE: {:.2f},LC Loss: {:.2f}, ULC Loss: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, epoch_loss.get_avg(), epoch_count_loss.get_avg(), epoch_ot_loss.get_avg(), epoch_tv_loss.get_avg(), epoch_rank_loss.get_avg(),
                epoch_consistensy_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), epoch_count_consistency_l.get_avg(), 
                epoch_count_consistency_ul.get_avg(), time.time() - epoch_start))
                
         
                
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, "{}_ckpt.tar".format(self.epoch))
        
        torch.save({"epoch": self.epoch, "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": model_state_dic}, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name, gauss_im in self.dataloaders["val"]:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(
                    lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    
                    crop_pred, _ = self.model(crop_imgs[gs:gt])
                    crop_pred = crop_pred[0]
                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (F.interpolate(crop_pred, size=(h1 * 4, w1 * 4),
                            mode="bilinear", align_corners=True) / 16 )

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info("Epoch {} Val, MSE: {:.2f}, MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch, mse, mae, time.time() - epoch_start ))


        model_state_dic = self.model.state_dict()
        print("Comaprison", mae,  self.best_mae)
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                "save best mse {:.2f} mae {:.2f} model epoch {}".format(
                    self.best_mse, self.best_mae, self.epoch))
                    
            print("Saving best model at {} epoch".format(self.epoch))
            model_path = os.path.join(
                self.save_dir, "best_model_mae-{:.2f}_epoch-{}.pth".format(
                    self.best_mae, self.epoch))
                    
            torch.save(model_state_dic, model_path)


if __name__ == "__main__":
    import torch
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

    





