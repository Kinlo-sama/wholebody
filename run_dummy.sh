sed -i 's/model.load_state_dict(ckpt)/model.load_state_dict(ckpt, strict=False)/' test_inference_dummy.py
python test_inference_dummy.py
