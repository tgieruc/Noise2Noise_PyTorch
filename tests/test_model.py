import os
import tempfile

import torch

from noise2noise import Dataset, Model, Net


def test_net_forward_shape():
    net = Net()
    x = torch.randn(2, 3, 32, 32)
    out = net(x)
    assert out.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {out.shape}"


def test_dataset_length_and_shape():
    data1 = torch.randn(10, 3, 32, 32)
    data2 = torch.randn(10, 3, 32, 32)
    ds = Dataset(data1, data2)
    assert len(ds) == 10
    item = ds[0]
    assert item.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {item.shape}"


def test_model_predict_range():
    model = Model()
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.float32)
    out = model.predict(x)
    assert out.shape == (4, 3, 32, 32)
    assert out.min() >= 0, f"Output min {out.min()} < 0"
    assert out.max() <= 255, f"Output max {out.max()} > 255"


def test_save_load_roundtrip():
    model = Model()
    x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.float32)
    out1 = model.predict(x)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        model2 = Model()
        model2.load_pretrained_model(path)
        out2 = model2.predict(x)
        assert torch.allclose(out1, out2), "Outputs differ after save/load"
    finally:
        os.unlink(path)


def test_device_consistency():
    model = Model()
    assert model.device is not None
    # Net should be on the same device
    param_device = next(model.net.parameters()).device
    assert str(param_device) == str(model.device)


if __name__ == "__main__":
    tests = [
        test_net_forward_shape,
        test_dataset_length_and_shape,
        test_model_predict_range,
        test_save_load_roundtrip,
        test_device_consistency,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
