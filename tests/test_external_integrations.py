import unittest
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.onnx_utils import export_to_onnx, load_onnx_session, predict_onnx
from src.huggingface_utils import HuggingFaceService, _check_hf_availability
from src.onnx_utils import _check_onnx_availability

class TestExternalIntegrations(unittest.TestCase):
    def setUp(self):
        # Create a simple scikit-learn model for testing export
        self.X = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
        self.y = np.random.randint(0, 2, 10)
        self.model = LogisticRegression()
        self.model.fit(self.X, self.y)
        self.onnx_path = "tests/test_model.onnx"
        if not os.path.exists("tests"):
            os.makedirs("tests")

    def test_onnx_export_and_inference(self):
        if not _check_onnx_availability():
            self.skipTest("ONNX stack not available in environment")

        import onnxruntime as ort

        # Test Export
        path = export_to_onnx(self.model, "flaml", "target", self.onnx_path, input_sample=self.X[:1])
        self.assertTrue(os.path.exists(path))
        
        # Test Load
        session = load_onnx_session(path)
        self.assertIsInstance(session, ort.InferenceSession)
        
        # Test Predict
        preds = predict_onnx(session, self.X[:1])
        self.assertEqual(len(preds), 1)
        # sklearn LogisticRegression output in ONNX is usually [label, probabilities]
        # Our utility returns outputs[0] which is labels

    def test_huggingface_service_init(self):
        if not _check_hf_availability():
            self.skipTest("huggingface_hub not available in environment")

        # We don't have a token for CI, so we just test initialization
        service = HuggingFaceService(token="dummy_token")
        self.assertEqual(service.token, "dummy_token")
        self.assertIsNotNone(service.api)

    def tearDown(self):
        if os.path.exists(self.onnx_path):
            os.remove(self.onnx_path)

if __name__ == '__main__':
    unittest.main()
