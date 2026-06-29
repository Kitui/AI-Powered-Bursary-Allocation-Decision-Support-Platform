import pandas as pd
import torch
import joblib

from app import (
    REQUIRED_UPLOAD_COLUMNS,
    allocation_reason,
    normalize_upload_dataframe,
    parse_number,
    state_features,
    validate_upload_dataframe,
)


def valid_upload_frame():
    return pd.DataFrame([{
        'Name': 'Jane Applicant',
        'Year of Application': '2025',
        'Gender': 'Female',
        'Ward': 'Matisi',
        'Academic Level': 'Secondary School',
        'Average Academic Performance': 'Good',
        'School': 'Example School',
        'Course of Study': 'Highschool',
        'Course Duration': '4',
        'Mode of Study': 'Boarding',
        'Year of Course Completion': '2027',
        'Amount Applied (Kshs)': '38,000',
        'Fee Balance': '40,000',
        'Family status.': 'Single Parent',
        'Care of': 'Mother',
        'Employment Type': 'Casual',
        'Is it a main Source of Income()': 'No',
        'Past Financial Support(NG-CDF)': 'No',
        'Past Financial Support(Others)': 'No',
        'If yes, specify how much.': '',
        'Last Received': '',
        'Individual Disability Status': 'No',
        'Parent/Guardian Disability Status': 'No',
        'Supportive documents Available': 'Yes',
        'Recommendation': 'Approve',
    }])


def test_parse_number_accepts_commas_and_empty_values():
    assert parse_number('38,000') == 38000
    assert parse_number('None') == 0
    assert parse_number('', 7) == 7


def test_upload_validation_accepts_normalized_project_csv_shape():
    df = normalize_upload_dataframe(valid_upload_frame())
    errors, warnings = validate_upload_dataframe(df)
    assert errors == []
    assert isinstance(warnings, list)
    for column in REQUIRED_UPLOAD_COLUMNS:
        assert column in df.columns


def test_upload_validation_reports_missing_columns():
    df = pd.DataFrame([{'Name': 'Only Name'}])
    errors, _ = validate_upload_dataframe(df)
    assert errors
    assert errors[0]['type'] == 'missing_columns'
    assert 'Amount Applied (Kshs)' in errors[0]['columns']


def test_upload_validation_reports_bad_amount_rows():
    df = normalize_upload_dataframe(valid_upload_frame())
    df.loc[0, 'Amount Applied (Kshs)'] = '-1'
    errors, _ = validate_upload_dataframe(df)
    assert any(error['type'] == 'invalid_amount_applied' for error in errors)


def test_ddpg_artifacts_match_state_contract():
    params = joblib.load('ddpg_agent_params.joblib')
    actor_state = torch.load('ddpg_actor.pth', map_location='cpu')
    critic_state = torch.load('ddpg_critic.pth', map_location='cpu')

    assert params['state_features'] == state_features
    assert params['state_dim'] == len(state_features)
    assert actor_state['layer1.weight'].shape[1] == len(state_features)
    assert critic_state['layer1.weight'].shape[1] == len(state_features) + 1


def test_allocation_reason_mentions_major_need_factors():
    row = normalize_upload_dataframe(valid_upload_frame()).iloc[0]
    reason = allocation_reason(row, score=0.91, allocation=1000)
    assert 'High predicted financial need' in reason
    assert 'Family vulnerability' in reason
    assert 'Supportive documents available' in reason
