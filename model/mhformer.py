import torch
import torch.nn as nn
from einops import rearrange
from model.module.trans import Transformer as Transformer_s
from model.module.trans_hypothesis import Transformer

# from model.testing_module.poseformer import PoseTransformer

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ###############################################################
        # self.poseformer  = PoseTransformer(num_frame=args.frames, num_joints=args.n_joints, in_chans=args.channel, num_heads=9)
        # """"
        # num_frame=27, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
        #          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
        #          drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None
        # """
        ################################################################

        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)
############################################################################################################## start
        ## original code:
        self.trans_auto_1 = Transformer_s(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.trans_auto_2 = Transformer_s(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.trans_auto_3 = Transformer_s(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        self.encoder_1 = nn.Sequential(nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1))
        self.encoder_2 = nn.Sequential(nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1))
        self.encoder_3 = nn.Sequential(nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1))

        self.encoder_score = nn.Sequential(nn.Conv1d(args.n_joints, args.channel, kernel_size=1))

############################################################################################################### end
        self.Transformer = Transformer(args.layers, args.channel*3, args.d_hid, length=args.frames)
        
        # self.fcn = nn.Sequential(
        #     nn.BatchNorm1d(args.channel*3, momentum=0.1),
        #     ####### A easy way to implement weighted mean
        #     nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        # )
############################################################################################################### start
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.channel*6, momentum=0.1),
            ####### A easy way to implement weighted mean
            nn.Conv1d(args.channel*6, 3*args.out_joints, kernel_size=1)

        #     nn.BatchNorm1d(args.channel*3, momentum=0.1),
        #     ####### A easy way to implement weighted mean
        #     nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )
############################################################################################################### end

        # borrow from poseformer
        # self.head = nn.Sequential(
        #     nn.LayerNorm(512*17),
        #     nn.Linear(512, 17)
        # )


    def forward(self, x):
        """
        Args
            B : Batch size
            F : Number of frames
            J : Number of joints
            C : Channel size
            x : concatenate the (x,y) coordinates of joints for each frame to 
                " X_bar = R ^ (J*2) * N "
            score: visibility score
                
        """
        B, F, J, C = x.shape
        ############################################################### start
        # Debug, To see x.shape
        # print("Debug to see x.shape(): \n\n", x.shape) # add vs's x.shape=[128, 9, 17, 3]
        
        # x_poseformer = rearrange(x, 'b f j c -> b c f j').contiguous()
        score = x[:, :, :, 2]
        score = rearrange(score, 'b f j -> b j f').contiguous()
        ############################################################### end
        x = x[..., :2]
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()


        ## Poseformer
        # x_poseformer = self.poseformer(x_poseformer)

        ## MHG
        x_1 = x   + self.trans_auto_1(self.norm_1(x))
        x_2 = x_1 + self.trans_auto_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.trans_auto_3(self.norm_3(x_2))
        
        ############################################################### start
        ## (1) 先試試直接把score concat 在 MHG output 的 Encoded feature， performance 是多少
        # x_1 = torch.cat([x_1, score], 1)
        # x_2 = torch.cat([x_2, score], 1)
        # x_3 = torch.cat([x_3, score], 1)

        ## (2) turn confidence score to embedding
        score = self.encoder_score(score) # [8 512 9]
        score = score.permute(0, 2, 1).contiguous() # [8 9 512]
        ############################################################### end
        
        
        ## Embedding
        x_1 = self.encoder_1(x_1) # encoded feature 【dim(2 * n_joint)】 會被 embedded 到 high-dimensional feature 【dim(channel)】 
        x_1 = x_1.permute(0, 2, 1).contiguous() 

        x_2 = self.encoder_2(x_2)
        x_2 = x_2.permute(0, 2, 1).contiguous()

        x_3 = self.encoder_3(x_3) 
        x_3 = x_3.permute(0, 2, 1).contiguous()

        ## SHR & CHI
        # x = self.Transformer(x_1, x_2, x_3) 
        x = self.Transformer(x_1, x_2, x_3, score) 


        ## Head
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x






