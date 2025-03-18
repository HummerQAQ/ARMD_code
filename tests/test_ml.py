import numpy as np
import pytest
from bson import ObjectId

from tests import model_algorithm, predict, train, train_predict_delete

"""
Implement your own test methods if you got something else to test!

Hints:
    1. Most of the time you want to test whether you can train and predict successfully. If that is the case,
       do it like the `test_air_passengers` test method below, where you put the csv files and the `meta.json`
       configuration file in the right directory, then provide that directory to `train_predict_delete` method.
"""


async def test_air_passengers():
    """
    Tests whether a simple dataset like `AirPassengers` can be trained and predicted upon.
    """
    await train_predict_delete("AirPassengers")


async def test_shanghai_car_license_plate_auction_price():
    """
    Tests whether a simple dataset with time groups like `Shanghai_Car_License_Plate_Auction_Price` can be
    trained and predicted upon.
    """
    await train_predict_delete("Shanghai_Car_License_Plate_Auction_Price")


async def test_target_pivot_handle_list_of_tuple():
    """
    Tests if the algorithm can pass issue DE-7241.
    The content of the time groups columns have numeric and string.
    """
    await train_predict_delete("target_pivot_handle_list_of_tuple")


async def test_predict_unseen_time_groups():
    """
    Tests whether a test dataset with unseen time groups can be
    trained and predicted upon.
    """
    await train_predict_delete("predict_unseen_time_groups")


async def test_predict_unexpected_date():
    """
    Tests whether a test dataset with unexpected date can be predicted.
    """
    await train_predict_delete("predict_unexpected_date")


async def test_no_time_group_some_invalid_time_series():
    """
    Test the condition of
        (1) No time group column
        (2) The last fold is invalid
    """
    name = "test_no_time_group_some_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # third fold is invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        assert not np.isnan(fold_scores[0])
        assert not np.isnan(fold_scores[1])
        assert np.isnan(fold_scores[2])

    response = await predict(name, model_id)
    assert response is not None


async def test_one_time_group_some_invalid_time_series():
    """
    Test the condition of
        (1) 1 time group column
        (2) All of the time groups in the last fold are invalid
    """
    name = "test_one_time_group_some_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # third fold is invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        assert not np.isnan(fold_scores[0])
        assert not np.isnan(fold_scores[1])
        assert np.isnan(fold_scores[2])

    response = await predict(name, model_id)
    assert response is not None


async def test_two_time_groups_some_invalid_time_series():
    """
    Test the condition of
        (1) 2 time group columns
        (2) All of the time groups in the last fold are invalid
    """
    name = "test_two_time_groups_some_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # third fold is invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        assert not np.isnan(fold_scores[0])
        assert not np.isnan(fold_scores[1])
        assert np.isnan(fold_scores[2])

    response = await predict(name, model_id)
    assert response is not None


async def test_no_time_group_no_invalid_time_series():
    """
    Test the condition of
        (1) No time group column
        (2) All folds are invalid
    """
    name = "test_no_time_group_no_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # All folds are invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        for fold_score in fold_scores:
            assert np.isnan(fold_score)

    response = await predict(name, model_id)
    assert response is not None


async def test_one_time_group_no_invalid_time_series():
    """
    Test the condition of
        (1) One time group column
        (2) All folds are invalid
        (3) Expected to raise a ValueError with a specific message regarding data shape issues, due to the data shortage to meet the derivation_window length.
    """
    name = "test_one_time_group_no_invalid_time_series"
    try:
        model_id = await train(name)
    except Exception as e:
        assert "cannot reshape array of size" in str(
            e
        ), f"Unexpected error message: {str(e)}"
    else:
        pytest.fail(
            f"Unexpected result: Successfully got model_id: {model_id} for dataset {name}."
        )


async def test_two_time_groups_no_invalid_time_series():
    """
    Test the condition of
        (1) Two time group columns
        (2) All folds are invalid
        (3) Expected to raise a ValueError with a specific message regarding data shape issues, due to the data shortage to meet the derivation_window length.
    """
    name = "test_two_time_groups_no_invalid_time_series"
    try:
        model_id = await train(name)
    except Exception as e:
        assert "cannot reshape array of size" in str(
            e
        ), f"Unexpected error message: {str(e)}"
    else:
        pytest.fail(
            f"Unexpected result: Successfully got model_id: {model_id} for dataset {name}."
        )


async def test_no_time_group_with_gap_some_invalid_time_series():
    """
    Test the condition of
        (1) No time group column
        (2) The gap is larger than 0 and is the cause of invalid time series
        (3) The last fold is invalid
    """
    name = "test_no_time_group_with_gap_some_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # The second and third fold are invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        assert not np.isnan(fold_scores[0])
        assert np.isnan(fold_scores[1])
        assert np.isnan(fold_scores[2])

    response = await predict(name, model_id)
    assert response is not None


async def test_one_time_group_with_gap_some_invalid_time_series():
    """
    Test the condition of
        (1) 1 time group column
        (2) The gap is larger than 0 and is the cause of invalid time series
        (3) The last fold is invalid
    """
    name = "test_one_time_group_with_gap_some_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # The second and third fold are invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        assert not np.isnan(fold_scores[0])
        assert np.isnan(
            fold_scores[1]
        )  # Encounter the condition where there is not enough data to train
        assert np.isnan(fold_scores[2])

    response = await predict(name, model_id)
    assert response is not None


async def test_two_time_groups_with_gap_some_invalid_time_series():
    """
    Test the condition of
        (1) 2 time group columns
        (2) The gap is larger than 0 and is the cause of invalid time series
        (3) The last fold is invalid
    """
    name = "test_two_time_groups_with_gap_some_invalid_time_series"
    model_id = await train(name)
    assert model_id
    model_json = model_algorithm.collection.find_one({"_id": ObjectId(model_id)})
    assert model_json is not None
    # Check if all the scores in an invalid fold are NAs
    # The second and third fold are invalid
    scores = model_json["attributes"]["cv_scores"]
    for fold_scores in scores.values():
        assert not np.isnan(fold_scores[0])
        assert np.isnan(
            fold_scores[1]
        )  # Encounter the condition where there is not enough data to train
        assert np.isnan(fold_scores[2])

    response = await predict(name, model_id)
    assert response is not None


async def test_two_time_groups_some_invalid_time_series_with_endo_features():
    """
    Tests the condition of
        (1) Irregular time series
        (2) 2 time group columns
        (3) Many endogenous features
    """
    name = "test_two_time_groups_some_invalid_time_series_with_endo_features"
    model_id = await train(name)
    assert model_id
