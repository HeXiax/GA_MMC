import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pre_train import DeepAE
from built_graph1 import AttentionLayer,GNNLayer
from built_graph1 import built_cos_knn
from built_graph1 import KNN_Att
from built_graph1 import Multi_head_attention



class IDMMC(nn.Module):
    def __init__(self, args, config, cptpath):
        super(IDMMC, self).__init__()
        self.args = args
        self.config = config

        assert config['img_hiddens'][-1] == config['txt_hiddens'][-1], \
            'Inconsistent latent dim!'

        self.latent_dim = config['img_hiddens'][-1]

        dicts = torch.load(cptpath)

        self.imgAE = DeepAE(input_dim=config['img_input_dim'],
                            hiddens=config['img_hiddens'],
                            batchnorm=config['batchnorm'])
        self.imgAE.load_state_dict(dicts['G1_state_dict'])

        self.txtAE = DeepAE(input_dim=config['txt_input_dim'],
                            hiddens=config['txt_hiddens'],
                            batchnorm=config['batchnorm'])
        self.txtAE.load_state_dict(dicts['G2_state_dict'])

        self.img2txt = DeepAE(input_dim=self.latent_dim,
                              hiddens=config['img2txt_hiddens'],
                              batchnorm=config['batchnorm'])
        self.img2txt.load_state_dict(dicts['G12_state_dict'])

        self.txt2img = DeepAE(input_dim=self.latent_dim,
                              hiddens=config['txt2img_hiddens'],
                              batchnorm=config['batchnorm'])
        self.txt2img.load_state_dict(dicts['G21_state_dict'])

        # self.img_att = AttentionLayer(self.latent_dim, self.latent_dim)
        # self.txt_att = AttentionLayer(self.latent_dim, self.latent_dim)

        self.img_att = Multi_head_attention(self.latent_dim, self.latent_dim)
        self.txt_att = Multi_head_attention(self.latent_dim, self.latent_dim)


        self.img_att1 = KNN_Att(self.latent_dim, self.latent_dim)
        self.txt_att1 = KNN_Att(self.latent_dim, self.latent_dim)

        self.Cross_att1 = AttentionLayer(self.latent_dim, self.latent_dim, Cross=True)
        self.Cross_att2 = AttentionLayer(self.latent_dim, self.latent_dim, Cross=True)

        self.gconv = GNNLayer(self.latent_dim, self.latent_dim)

        self.img_gnn = GNNLayer(self.latent_dim, self.latent_dim)
        self.txt_gnn = GNNLayer(self.latent_dim, self.latent_dim)

        self.fc1 = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)



    def forward(self, feats, modalitys, k1, k2, k3):
        img_idx = modalitys.view(-1) == 0
        txt_idx = modalitys.view(-1) == 1
        img_feats = feats[img_idx]
        txt_feats = feats[txt_idx]
        c_fea_img = torch.zeros(feats.shape[0],self.latent_dim).cuda()
        c_fea_txt = torch.zeros(feats.shape[0], self.latent_dim).cuda()


        imgs_recon, imgs_latent = self.imgAE(img_feats)
        txts_recon, txts_latent = self.txtAE(txt_feats)

        img2txt_recon, _ = self.img2txt(imgs_latent)
        img_latent_recon, _ = self.txt2img(img2txt_recon)
        img_feats_recon = self.imgAE.decoder(img_latent_recon)

        txt2img_recon, _ = self.txt2img(txts_latent)
        txt_latent_recon, _ = self.img2txt(txt2img_recon)
        txt_feats_recon = self.txtAE.decoder(txt_latent_recon)

        c_fea_img[img_idx] = imgs_latent
        #c_fea_img[txt_idx] = txt2img_recon

        c_fea_txt[txt_idx] = txts_latent
        #c_fea_txt[img_idx] = img2txt_recon

        #intra_modal complete
        txt_img_adj = self.img_att(txt2img_recon, imgs_latent, k1)
        g_fea_img = torch.mm(txt_img_adj, imgs_latent)
        c_fea_img[txt_idx] = g_fea_img

        img_txt_adj = self.txt_att(img2txt_recon, txts_latent, k1)
        g_fea_txt = torch.mm(img_txt_adj, txts_latent)
        c_fea_txt[img_idx] = g_fea_txt


        # intra_modal message propogate
        img_adj = built_cos_knn(c_fea_img, k2)
        c_fea_img = torch.mm(img_adj, c_fea_img)

        txt_adj = built_cos_knn(c_fea_txt, k2)
        c_fea_txt = torch.mm(txt_adj, c_fea_txt)
        # #
        # #
        #inter_modal complete
        S1, S2 = self.img_att1(c_fea_img, c_fea_txt, k3)
        c_txt2img_fea = torch.mm(S1, c_fea_txt)
        h_img_feat = c_fea_img + c_txt2img_fea


        c_img2txt_fea = torch.mm(S2, c_fea_img)
        h_txt_feat = c_fea_txt + c_img2txt_fea



        # # intra_modal complete 2
        # img_adj1 = built_cos_knn(h_img_feat, k2)
        # h_img_feat = torch.mm(img_adj1, h_img_feat)
        #
        # txt_adj1 = built_cos_knn(h_txt_feat, k2)
        # h_txt_feat = torch.mm(txt_adj1, h_txt_feat)


        # #inter_modal complete 2
        # S1, S2 = self.txt_att1(h_img_feat, h_txt_feat, k3)
        # c_txt2img_fea1 = torch.mm(S1, h_txt_feat)
        # h_img_feat = h_img_feat + c_txt2img_fea1
        #
        # c_img2txt_fea1 = torch.mm(S2, h_img_feat)
        # h_txt_feat = h_txt_feat + c_img2txt_fea1

        #h_fuse = torch.cat((h_img_feat, h_txt_feat), dim=1)
        # h1 = F.relu(self.fc1(h))
        # h_fuse = self.fc2(h1)


        #h_fuse = h_img_feat + h_txt_feat
        #h_fuse = c_fea_img + c_fea_txt
        #h_fuse = torch.cat((c_fea_img, c_fea_txt), dim=1)
        h_fuse = torch.cat((h_img_feat, h_txt_feat), dim=1)

        h_img_all = [img_feats, imgs_recon, imgs_latent, img_latent_recon, img_feats_recon]
        h_txt_all = [txt_feats, txts_recon, txts_latent, txt_latent_recon, txt_feats_recon]

        return h_img_all, h_txt_all, h_fuse, c_fea_img, c_fea_txt





































