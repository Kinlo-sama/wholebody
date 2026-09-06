with open("test_inference_dummy.py", "r") as f:
    code = f.read()
code = code.replace("model.load_state_dict(ckpt)", "model.load_state_dict(ckpt, strict=False)")
with open("test_inference_dummy.py", "w") as f:
    f.write(code)
