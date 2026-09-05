from mmengine.config import Config
cfg = Config.fromfile('/Users/kinlo/.gemini/antigravity/scratch/wholebody/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py')
print("backbone use_depthwise:", cfg.model.backbone.get('use_depthwise', 'Not specified (False)'))
print("neck use_depthwise:", cfg.model.neck.get('use_depthwise', 'Not specified (False)'))
