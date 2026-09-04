with open('wholebody/evaluation/coco_metric.py', 'r') as f:
    content = f.read()

old_eval = """            # Run evaluation natively for keypoints_wholebody
            coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='keypoints_wholebody')
            
            # Inject the 133 sigmas (xtcocotools still requires us to pass them explicitly)
            coco_eval.params.kpt_oks_sigmas = sigmas
            
            coco_eval.evaluate()"""

new_eval = """            # Run evaluation natively for keypoints_wholebody
            # xtcocotools COCOeval takes sigmas and use_area in the constructor!
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints_wholebody', sigmas, use_area=True)
            coco_eval.params.useSegm = None
            
            coco_eval.evaluate()"""

content = content.replace(old_eval, new_eval)

with open('wholebody/evaluation/coco_metric.py', 'w') as f:
    f.write(content)

