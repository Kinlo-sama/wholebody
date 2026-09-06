import re
with open('wholebody/models/distillers/pose_estimator_distiller.py', 'r') as f:
    content = f.read()

content = content.replace('def loss(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> Dict[str, torch.Tensor]:', 'def forward_train(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> Dict[str, torch.Tensor]:')

with open('wholebody/models/distillers/pose_estimator_distiller.py', 'w') as f:
    f.write(content)

with open('test_dummy_distiller.py', 'r') as f:
    test_content = f.read()

test_content = test_content.replace('losses = distiller.loss(x, data_samples)', 'losses = distiller.forward_train(x, data_samples)')

with open('test_dummy_distiller.py', 'w') as f:
    f.write(test_content)
