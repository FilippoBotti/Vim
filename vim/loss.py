import torch.nn.functional as F
from torch import nn
from utils import normal, calc_mean_std

class VGGLoss(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self, encoder, args):
        super().__init__()

        self.device = args.device
        self.encoder = encoder.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.args = args

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, input, style, output):
        # ### features used to calcate loss 
        content_feats = self.encoder(input)
        style_feats = self.encoder(style)
        out_feats = self.encoder(output)
        
        loss_c = self.calc_content_loss(normal(out_feats), normal(content_feats))
        loss_s = self.calc_style_loss(out_feats, style_feats)
            
        loss = self.args.content_weight * loss_c + self.args.style_weight * loss_s
        # return Ics    #test 
        return loss