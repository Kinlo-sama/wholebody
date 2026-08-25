import unittest
import numpy as np
import torch
from wholebody.codecs.heatmap import MSRAHeatmapCodec
from wholebody.codecs.regression import RegressionCodec


class TestCodecs(unittest.TestCase):

    def test_msra_heatmap_codec_encode_decode(self):
        codec = MSRAHeatmapCodec(input_size=(256, 192), heatmap_size=(64, 48), sigma=2.0)
        # Ground truth keypoints in 256x192 crop space
        kpts = np.array([[96.0, 128.0], [50.0, 60.0]], dtype=np.float32)
        vis = np.array([2.0, 2.0], dtype=np.float32)

        encoded = codec.encode(kpts, vis)
        self.assertEqual(encoded["heatmaps"].shape, (2, 64, 48))
        self.assertEqual(encoded["keypoint_weights"].shape, (2,))

        # Peak of heatmap 0 should be at feat_y = 128*(64/256)=32, feat_x = 96*(48/192)=24
        hm0 = encoded["heatmaps"][0]
        max_idx = np.unravel_index(np.argmax(hm0), hm0.shape)
        self.assertEqual(max_idx, (32, 24))

        # Decode batch
        batch_hm = torch.from_numpy(encoded["heatmaps"]).unsqueeze(0)  # (1, 2, 64, 48)
        decoded_kpts, scores = codec.decode(batch_hm)

        self.assertEqual(decoded_kpts.shape, (1, 2, 2))
        np.testing.assert_allclose(decoded_kpts[0], kpts, atol=2.5)
        self.assertGreater(scores[0, 0], 0.9)

    def test_regression_codec(self):
        codec = RegressionCodec(input_size=(256, 192))
        kpts = np.array([[192.0, 128.0]], dtype=np.float32)
        encoded = codec.encode(kpts)
        np.testing.assert_allclose(encoded["target_coords"], [[1.0, 0.5]], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
