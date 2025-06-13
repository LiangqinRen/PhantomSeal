from models.models import create_model
from evaluate import Utility, Effectiveness, AIEditing, Cloak

import torch
import torch.nn.functional as F
from argparse import Namespace
from torch import tensor
from types import MethodType


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        # self.test_options = Namespace(
        #     gpu_ids=[0],
        #     isTrain=False,
        #     checkpoints_dir="third_party/SimSwap/checkpoints",
        #     name="people",
        #     resize_or_crop="scale_width",
        #     crop_size=224,
        #     Arc_path="third_party/SimSwap/arcface_model/arcface_checkpoint.tar",
        #     which_epoch="latest",
        #     verbose=False,
        # )
        self.test_options = Namespace(
            name="people",
            gpu_ids=[0],
            checkpoints_dir="third_party/SimSwap/checkpoints",
            norm="batch",
            use_dropout=False,
            data_type=32,
            verbose=False,
            fp16=False,
            local_rank=0,
            isTrain=False,
            batchSize=8,
            loadSize=1024,
            fineSize=512,
            label_nc=0,
            input_nc=3,
            output_nc=3,
            dataroot="./datasets/cityscapes/",
            resize_or_crop="scale_width",
            serial_batches=False,
            no_flip=False,
            nThreads=2,
            max_dataset_size=2**1024,
            display_winsize=512,
            tf_log=False,
            netG="global",
            latent_size=512,
            ngf=64,
            n_downsample_global=3,
            n_blocks_global=6,
            n_blocks_local=3,
            n_local_enhancers=1,
            niter_fix_global=0,
            no_instance=False,
            instance_feat=False,
            label_feat=False,
            feat_num=3,
            load_features=False,
            n_downsample_E=4,
            nef=16,
            n_clusters=10,
            image_size=224,
            norm_G="spectralspadesyncbatch3x3",
            semantic_nc=3,
            ntest=2**1024,
            results_dir="./results/",
            aspect_ratio=1.0,
            phase="test",
            which_epoch="latest",
            how_many=50,
            cluster_path="features_clustered_010.npy",
            use_encoded_image=False,
            export_onnx=None,
            engine=None,
            onnx=None,
            Arc_path="third_party/SimSwap/arcface_model/arcface_checkpoint.tar",
            pic_a_path="G:/swap_data/ID/elon-musk-hero-image.jpeg",
            pic_b_path="./demo_file/multi_people.jpg",
            pic_specific_path="./crop_224/zrf.jpg",
            multisepcific_dir="./demo_file/multispecific",
            video_path="G:/swap_data/video/HSB_Demo_Trim.mp4",
            temp_path="./temp_results",
            output_path="./output/",
            id_thres=0.03,
            no_simswaplogo=False,
            use_mask=False,
            crop_size=224,
        )

        self.target = create_model(self.test_options)

        def encoder(self, input):
            x = input

            x = self.first_layer(x)
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            if self.deep:
                x = self.down4(x)

            return x

        self.target.netG.encoder = MethodType(encoder, self.target.netG)

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.aiediting = AIEditing(logger, config)
        self.cloak = Cloak(logger, config, self.effectiveness)

    def _get_imgs_identity(self, imgs: tensor) -> tensor:
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.target.netArc(imgs_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()
