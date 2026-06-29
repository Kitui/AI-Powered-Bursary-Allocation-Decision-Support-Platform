import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

from app import (
    Actor,
    build_state_data,
    lgb_model,
    meta_model,
    model_input_features,
    normalize_scores,
    normalize_upload_dataframe,
    parse_int,
    parse_number,
    pca,
    pca_features,
    rf_model,
    state_features,
    validate_upload_dataframe,
    xgb_model,
)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


def build_model_frame(upload_df):
    rows = []
    for _, row in upload_df.iterrows():
        mapped = {}
        year_of_application = parse_int(row.get('Year of Application', 2025), 2025)
        academic_level = row.get('Academic Level', 'Secondary School')
        mapped['acad_level_University'] = 1 if academic_level == 'University' else 0
        acad_perf = row.get('Average Academic Performance', 'Poor')
        mapped['acad_perf_Good'] = 1 if acad_perf == 'Good' else 0
        mapped['acad_perf_Fair'] = 1 if acad_perf == 'Fair' else 0
        mapped['acad_perf_Poor'] = 1 if acad_perf == 'Poor' else 0
        course_duration = parse_number(row.get('Course Duration', 0))
        mapped['course_dur'] = course_duration
        mode_of_study = row.get('Mode of Study', 'Boarding')
        mapped['study_mode_Boarding'] = 1 if mode_of_study in ['Boarding', 'Bording'] else 0
        mapped['study_mode_Day scholar'] = 1 if mode_of_study == 'Day scholar' else 0
        mapped['study_mode_Government Sponsored'] = 1 if mode_of_study == 'Government Sponsored' else 0
        mapped['study_mode_Self sponsored'] = 1 if mode_of_study == 'Self sponsored' else 0
        expected_completion_year = parse_int(row.get('Year of Course Completion', 2025), 2025)
        mapped['exp_completion'] = expected_completion_year
        mapped['amt_applied'] = parse_number(row.get('Amount Applied (Kshs)', 0))
        mapped['fee_balance'] = parse_number(row.get('Fee Balance', 0))
        family_status = row.get('Family status', 'Other')
        mapped['family_status_Total Orphan'] = 1 if family_status == 'Total Orphan' else 0
        mapped['family_status_Partial orphan'] = 1 if family_status == 'Partial orphan' else 0
        mapped['family_status_Single Parent'] = 1 if family_status == 'Single Parent' else 0
        caregiver = row.get('Care of', 'Mother')
        mapped['caregiver_Guardian'] = 1 if caregiver == 'Guardian' else 0
        mapped['caregiver_Father'] = 1 if caregiver == 'Father' else 0
        mapped['caregiver_Mother'] = 1 if caregiver == 'Mother' else 0
        emp_type = row.get('Employment Type', 'Unknown')
        mapped['emp_type_Contractual'] = 1 if emp_type == 'Contractual' else 0
        mapped['emp_type_Parmanent'] = 1 if emp_type == 'Permanent' else 0
        mapped['emp_type_Retired'] = 1 if emp_type == 'Retired' else 0
        mapped['emp_type_Self Employed'] = 1 if emp_type == 'Self Employed' else 0
        mapped['emp_type_Unknown'] = 1 if emp_type == 'Unknown' else 0
        mapped['past_ngcdf_Yes'] = 1 if row.get('Past Financial Support(NG-CDF)', 'No') == 'Yes' else 0
        mapped['other_support_Yes'] = 1 if row.get('Past Financial Support(Others)', 'No') == 'Yes' else 0
        support_amt = parse_number(row.get('If yes, specify how much.', 0))
        mapped['support_amt'] = support_amt
        last_support_year = parse_int(row.get('Last Received', 0))
        mapped['past_support_impact'] = max(0, 1 - (2025 - last_support_year) / 10) if last_support_year > 0 else 0
        mapped['ind_disability_Yes'] = 1 if row.get('Individual Disability Status', 'No') == 'Yes' else 0
        pg_disability = row.get('Parent/Guardian Disability Status', 'No')
        mapped['pg_disability_No'] = 1 if pg_disability == 'No' else 0
        mapped['pg_disability_Yes'] = 1 if pg_disability == 'Yes' else 0
        mapped['pg_disability_Chronic Disease'] = 1 if pg_disability == 'Chronic Disease' else 0
        mapped['pg_disability_Both'] = 1 if pg_disability == 'Both' else 0
        docs_available = row.get('Supportive documents Available', 'No')
        mapped['docs_available_No'] = 1 if docs_available == 'No' else 0
        mapped['docs_available_Partial'] = 1 if docs_available == 'Partial' else 0
        mapped['docs_available_Yes'] = 1 if docs_available == 'Yes' else 0
        mapped['vuln_index'] = (
            mapped['ind_disability_Yes'] +
            mapped['pg_disability_Yes'] +
            mapped['pg_disability_Chronic Disease'] +
            mapped['pg_disability_Both'] +
            mapped['family_status_Total Orphan'] +
            mapped['family_status_Partial orphan'] +
            mapped['family_status_Single Parent']
        ) / 7.0
        years_remaining = max(0, expected_completion_year - year_of_application)
        mapped['years_remaining'] = years_remaining
        mapped['course_completion_ratio'] = max(0, min(1, (course_duration - years_remaining) / course_duration)) if course_duration > 0 else 0
        mapped['log_financial_burden'] = np.log1p(mapped['fee_balance']) / np.log1p(1000000) if mapped['fee_balance'] > 0 else 0
        mapped['remaining_fee'] = max(0, mapped['fee_balance'] - support_amt)
        rows.append(mapped)

    input_df = pd.DataFrame(rows)
    input_df['vulnerability_score'] = (
        3 * input_df.get('ind_disability_Yes', 0) +
        2 * input_df.get('pg_disability_Yes', 0) +
        2 * input_df.get('pg_disability_Chronic Disease', 0) +
        4 * input_df.get('pg_disability_Both', 0) +
        3 * input_df.get('family_status_Total Orphan', 0) +
        2 * input_df.get('family_status_Partial orphan', 0) +
        input_df.get('family_status_Single Parent', 0) +
        (1 - input_df['past_ngcdf_Yes']) +
        np.log1p(input_df['remaining_fee']) / np.log1p(input_df['remaining_fee'].max()) * 5 +
        (input_df['course_completion_ratio'] / input_df['course_completion_ratio'].max()) * 2 +
        (1 - input_df['course_dur'] / 4) * 2 +
        input_df.get('docs_available_No', 0) +
        2 * input_df.get('emp_type_Contractual', 0) +
        input_df.get('emp_type_Unknown', 0) +
        input_df.get('study_mode_Boarding', 0) +
        2 * input_df.get('acad_level_University', 0)
    ).fillna(0)
    max_vuln = input_df['vulnerability_score'].max()
    if max_vuln > 0:
        input_df['vulnerability_score'] = input_df['vulnerability_score'] / max_vuln * 10

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    for feature in ['remaining_fee', 'vulnerability_score', 'log_financial_burden']:
        input_df[f'{feature}_transformed'] = pt.fit_transform(input_df[[feature]])

    features_for_score = [
        'remaining_fee_transformed',
        'vulnerability_score_transformed',
        'log_financial_burden_transformed',
        'course_completion_ratio',
        'past_support_impact',
    ]
    input_df[features_for_score] = StandardScaler().fit_transform(input_df[features_for_score])
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(input_df[features_for_score].fillna(0), input_df['amt_applied'].fillna(input_df['amt_applied'].median()))
    weights = pd.Series(ridge.coef_, index=features_for_score).to_dict()
    engineered_score = sum(weights[feature] * input_df[feature] for feature in features_for_score)
    engineered_score += 0.1 * input_df['remaining_fee_transformed'] * input_df['vulnerability_score_transformed']
    engineered_score = expit((engineered_score - engineered_score.mean()) / engineered_score.std())
    input_df['financial_need_score'] = 100 * (engineered_score - engineered_score.min()) / (engineered_score.max() - engineered_score.min())
    input_df['financial_need_score'] = input_df['financial_need_score'].clip(lower=1, upper=99)

    pca_df = pd.DataFrame(np.nan_to_num(pca.transform(input_df[pca_features]), nan=0.0), columns=['PC1', 'PC2', 'PC3', 'PC4'])
    model_frame = pd.concat([
        input_df[[
            'amt_applied',
            'vulnerability_score',
            'course_completion_ratio',
            'acad_perf_Good',
            'past_support_impact',
            'log_financial_burden',
            'remaining_fee',
            'remaining_fee_transformed',
            'log_financial_burden_transformed',
        ]],
        pca_df,
    ], axis=1)

    model_input = MinMaxScaler().fit_transform(model_frame[model_input_features])
    rf_pred = np.nan_to_num(rf_model.predict(model_input), nan=0.0)
    xgb_pred = np.nan_to_num(xgb_model.predict(model_input), nan=0.0)
    lgb_pred = np.nan_to_num(lgb_model.predict(model_input), nan=0.0)
    predicted_scores = np.clip(np.nan_to_num(meta_model.predict(np.column_stack((rf_pred, xgb_pred, lgb_pred))), nan=0.0), 0, 1)
    if float(np.std(predicted_scores)) < 0.001 or float(np.mean(predicted_scores >= 0.98)) > 0.5:
        predicted_scores = normalize_scores(input_df['financial_need_score'].values / 100.0)
    else:
        predicted_scores = normalize_scores(expit(StandardScaler().fit_transform(predicted_scores.reshape(-1, 1)).flatten()))
    model_frame['predicted_financial_need_score'] = predicted_scores
    return model_frame


def train_policy(states):
    torch.manual_seed(42)
    state_dim = len(state_features)
    action_dim = 1
    actor = Actor(state_dim, action_dim, 1.0)
    critic = Critic(state_dim, action_dim)

    need = states[:, 0]
    vulnerability = states[:, 1]
    amount = states[:, 2]
    pca_context = states[:, 3:].mean(axis=1)
    priority = 0.55 * need + 0.25 * vulnerability + 0.10 * amount + 0.10 * pca_context
    target_action = np.clip(0.2 + 0.8 * priority, 0.2, 1.0).astype(np.float32).reshape(-1, 1)

    state_tensor = torch.from_numpy(states.astype(np.float32))
    target_tensor = torch.from_numpy(target_action)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    for _ in range(220):
        order = torch.randperm(state_tensor.size(0))
        for start in range(0, state_tensor.size(0), 256):
            idx = order[start:start + 256]
            loss = F.mse_loss(actor(state_tensor[idx]), target_tensor[idx])
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()

    with torch.no_grad():
        actions = actor(state_tensor)
    for _ in range(80):
        order = torch.randperm(state_tensor.size(0))
        for start in range(0, state_tensor.size(0), 256):
            idx = order[start:start + 256]
            loss = F.mse_loss(critic(state_tensor[idx], actions[idx]), target_tensor[idx])
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

    return actor, critic


def main():
    upload_df = normalize_upload_dataframe(pd.read_csv('Bursary.csv', encoding='utf-8-sig'))
    errors, warnings = validate_upload_dataframe(upload_df)
    if errors:
        raise ValueError(json.dumps(errors, indent=2))
    if warnings:
        print(json.dumps(warnings[:3], indent=2))

    model_frame = build_model_frame(upload_df)
    states = build_state_data(model_frame[state_features])
    actor, critic = train_policy(states)
    torch.save(actor.state_dict(), 'ddpg_actor.pth')
    torch.save(critic.state_dict(), 'ddpg_critic.pth')
    params = {
        'state_dim': len(state_features),
        'action_dim': 1,
        'max_action': 1.0,
        'gamma': 0.99,
        'tau': 0.0005,
        'batch_size': 64,
        'epsilon': 0.01,
        'epsilon_decay': 0.995,
        'min_epsilon': 0.01,
        'train_frequency': 100,
        'state_features': state_features,
        'training_source': 'Bursary.csv',
        'policy_target': '0.2 + 0.8 * (0.55*need + 0.25*vulnerability + 0.10*amount + 0.10*pca_context)',
    }
    joblib.dump(params, 'ddpg_agent_params.joblib')
    print(f'Retrained DDPG policy artifacts with {states.shape[0]} real applicant states and {states.shape[1]} features.')


if __name__ == '__main__':
    main()
