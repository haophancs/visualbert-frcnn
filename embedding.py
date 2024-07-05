import torch
from tqdm import tqdm

from utils import Config
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess


class FRCNNEmbedding:
    def __init__(self, pretrained_name="unc-nlp/frcnn-vg-finetuned", embed_dim=50, device="cpu"):
        self.pretrained_name = pretrained_name
        self.embed_dim = embed_dim
        self.device = device
        
        self.frcnn_cfg = Config.from_pretrained(self.pretrained_name)
        self.frcnn_cfg.MODEL.DEVICE = self.device
        self.frcnn_cfg.min_detections = self.embed_dim
        self.frcnn_cfg.max_detections = self.embed_dim
    
        self.frcnn = GeneralizedRCNN.from_pretrained(self.pretrained_name, config=self.frcnn_cfg)
        self.image_preprocess = Preprocess(self.frcnn_cfg)
    
    def __call__(self, image_paths, return_tensor="pt"):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        feature_list = []
        for image_path in tqdm(image_paths):
            images, sizes, scales_yx = self.image_preprocess(image_path)
            try:
                output_dict = self.frcnn(
                    images,
                    sizes,
                    scales_yx=scales_yx,
                    padding="max_detections",
                    max_detections=self.frcnn_cfg.max_detections,
                    return_tensors="pt",
                )
                feature = output_dict.get("roi_features").detach().cpu()
            except Exception as err:
                feature = torch.randn([1, self.embed_dim, 2048])
                print(f"Warning: failed to embed image {image_path}. Caused by {str(err)}")

            feature_list.append(feature)
        
        features = torch.vstack(feature_list)
        if return_tensor == "pt":
            return features
        return features.numpy()