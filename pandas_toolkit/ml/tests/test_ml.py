import unittest

import pandas as pd

import pandas_toolkit.ml


class TestStandardScaler(unittest.TestCase):
    def test_standard_scaler_accessor_usage(self):
        df = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])

        actual_s = df.ml.standard_scaler(column="x")

        expected_s = pd.Series([-1, 1])

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)


class TestTrainValidationSplit(unittest.TestCase):
    def test_with_sensible_is_validation_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=2 / 3, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])
        expected_df_validation = pd.DataFrame({"x": [2], "y": [2]}, index=[2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)

    def test_with_not_perfectly_divisible_is_validation_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=0.5, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])
        expected_df_validation = pd.DataFrame({"x": [2], "y": [2]}, index=[2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)

    def test_with_is_validation_frac_eq_1(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=1, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        expected_df_validation = pd.DataFrame({"x": [], "y": []}, dtype="int64")

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)

    def test_with_is_validation_frac_eq_0(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=0, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [], "y": []}, dtype="int64")
        expected_df_validation = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)
