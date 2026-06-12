from xsvid_qwen3vl.dataset import Category, GroundTruth
from xsvid_qwen3vl.metrics import evaluate_map


def test_perfect_prediction_ap50_is_100():
    cats = [Category(1, "car")]
    gts = [GroundTruth(image_id=1, category_id=1, bbox=[10, 10, 20, 20], area=400, id=1)]
    preds = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    result = evaluate_map(gts, preds, cats)
    assert abs(result["summary"]["AP50"] - 100.0) < 1e-6
    assert abs(result["summary"]["mAP_50_95"] - 100.0) < 1e-6


def test_wrong_class_ap50_is_0():
    cats = [Category(1, "car"), Category(2, "person")]
    gts = [GroundTruth(image_id=1, category_id=1, bbox=[10, 10, 20, 20], area=400, id=1)]
    preds = [{"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.9}]
    result = evaluate_map(gts, preds, cats)
    assert result["summary"]["AP50"] == 0.0
