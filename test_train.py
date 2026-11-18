
from train import make_dataset, train_model, FEATURES

def test_dataset_has_required_columns():
    df = make_dataset()
    assert all(col in df.columns for col in FEATURES + ["passed"])

def test_target_contains_two_classes():
    df = make_dataset()
    assert set(df["passed"].unique()) == {0, 1}

def test_training_returns_predictable_model():
    model = train_model()
    assert hasattr(model, "predict")

def test_predictions_have_expected_count_and_values():
    model = train_model()
    df = make_dataset()

    predictions = model.predict(df[FEATURES].iloc[:3])

    assert len(predictions) == 3
    assert set(predictions).issubset({0, 1})
