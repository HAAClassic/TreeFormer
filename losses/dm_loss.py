import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.consistency_loss import *
from losses.ot_loss import OT_Loss

class DMLoss(nn.Module):
    def __init__(self):
        super(DMLoss, self).__init__()
        self.DMLoss = 0.0
        self.losses = {}

    def forward(self, results, points, gt_discrete):
        self.DMLoss = 0.0
        self.losses = {}
          
        if results is None:
            self.DMLoss = 0.0
        elif isinstance(results, list) and len(results) > 0:
            count = 0
            for i in range(len(results[0])):
                with torch.set_grad_enabled(False):
                    preds_mean = (results[0][i])/len(results[0][0][0])
                    
                for j in range(len(results)):
                    var_sel = softmax_kl_loss(results[j][i], preds_mean)
                    exp_var = torch.exp(-var_sel)
                    consistency_dist = (preds_mean - results[j][i]) ** 2
                    temploss = (torch.mean(consistency_dist * exp_var) /(exp_var + 1e-8) + var_sel)
                   
                    self.losses.update({'unlabel_{}_loss'.format(str(i+1)): temploss})
                    self.DMLoss += temploss
                    
                    # Compute counting loss.
                    count_loss = self.mae(outputs_L[0].sum(1).sum(1).sum(1),
                        torch.from_numpy(gd_count).float().to(self.device))*self.args.reg
                    epoch_count_loss.update(count_loss.item(), N)
                    
                    # Compute OT loss.
                    ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs_L[0], points)
                        
                    ot_loss = ot_loss * self.args.ot
                    ot_obj_value = ot_obj_value * self.args.ot
                    epoch_ot_loss.update(ot_loss.item(), N)
                    epoch_ot_obj_value.update(ot_obj_value.item(), N)
                    epoch_wd.update(wd, N)
                    
                    gd_count_tensor = (torch.from_numpy(gd_count).float()
                        .to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3))
                        
                    gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                    tv_loss = (self.tvloss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1)* 
                        torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.tv
                    epoch_tv_loss.update(tv_loss.item(), N)
                
                    count += 1
            if count > 0:
                self.multiconloss = self.multiconloss / count

                
        return self.multiconloss

