import pandas as pd
import pytest

from telecom_churn.services.customer_validation import (
    validate_customer_data,
    validate_known_columns,
)


def test_valid_customer_data():
    df = pd.DataFrame(
        {
            "TenureMonths": [12],
            "MonthlyCharges": [90.0],
            "SupportTickets": [2],
            "Complaints": [0],
            "NPS": [6],
            "CSAT": [3],
        }
    )

    validate_customer_data(df)


def test_invalid_nps():
    df = pd.DataFrame(
        {
            "NPS": [11],
        }
    )

    with pytest.raises(
        ValueError,
        match="NPS.*menor o igual a 10",
    ):
        validate_customer_data(df)


def test_invalid_csat():
    df = pd.DataFrame(
        {
            "CSAT": [6],
        }
    )

    with pytest.raises(
        ValueError,
        match="CSAT.*menor o igual a 5",
    ):
        validate_customer_data(df)


def test_negative_complaints():
    df = pd.DataFrame(
        {
            "Complaints": [-1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Complaints.*mayor o igual a 0",
    ):
        validate_customer_data(df)


def test_invalid_numeric_value():
    df = pd.DataFrame(
        {
            "MonthlyCharges": ["texto"],
        }
    )

    with pytest.raises(
        ValueError,
        match="MonthlyCharges.*valores no numéricos",
    ):
        validate_customer_data(df)


def test_unknown_columns():
    df = pd.DataFrame(
        {
            "TenureMonths": [12],
            "ColumnaExtra": ["dato"],
        }
    )

    allowed_columns = {
        "TenureMonths",
    }

    with pytest.raises(
        ValueError,
        match="columnas no reconocidas",
    ):
        validate_known_columns(
            df,
            allowed_columns,
        )