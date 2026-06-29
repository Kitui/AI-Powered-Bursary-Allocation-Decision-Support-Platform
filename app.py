from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from sqlalchemy import inspect, or_
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.linear_model import Ridge
import os
from datetime import datetime
import json
import logging
import time
import random
from scipy.special import expit  # Sigmoid function for transformations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///bursary.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
db = SQLAlchemy(app)

# Initialize Flask-Login and Bcrypt
login_manager = LoginManager(app)
login_manager.login_view = 'login'
bcrypt = Bcrypt(app)

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='applicant')  # 'applicant', 'admin', 'stakeholder'
    name = db.Column(db.String(100))
    gender = db.Column(db.String(20))
    ward = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('applications', lazy='dynamic'))
    allocations = db.relationship('Allocation', backref='application', lazy='dynamic')
    name = db.Column(db.String(100))
    academic_level = db.Column(db.String(50))
    year_of_application = db.Column(db.Integer)
    average_academic_performance = db.Column(db.String(20))
    school = db.Column(db.String(100))
    course_of_study = db.Column(db.String(100))
    course_duration = db.Column(db.Integer)
    mode_of_study = db.Column(db.String(50))
    expected_year_of_completion = db.Column(db.Integer)
    amount_applied = db.Column(db.Float)
    fee_balance = db.Column(db.Float)
    family_status = db.Column(db.String(50))
    care_of = db.Column(db.String(50))
    employment_type = db.Column(db.String(50))
    is_main_source_of_income = db.Column(db.String(10))
    past_ngcdf_support = db.Column(db.String(10))
    other_support = db.Column(db.String(10))
    other_support_amount = db.Column(db.Float)
    last_support_year = db.Column(db.Integer)
    individual_disability = db.Column(db.String(10))
    parent_disability = db.Column(db.String(50))
    documents_available = db.Column(db.String(20))
    recommendation = db.Column(db.String(20))
    status = db.Column(db.String(20), default='pending')
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

class Allocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    application_id = db.Column(db.Integer, db.ForeignKey('application.id'), nullable=False)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)  # Reference Batch model
    financial_need_score = db.Column(db.Float)
    allocated_amount = db.Column(db.Float)
    allocated_at = db.Column(db.DateTime, default=datetime.utcnow)
    batch = db.relationship('Batch', backref=db.backref('allocations', lazy=True))  # Relationship to Batch

class Disbursement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    allocation_id = db.Column(db.Integer, db.ForeignKey('allocation.id'), nullable=False)
    amount = db.Column(db.Float)
    status = db.Column(db.String(20), default='pending')
    disbursed_at = db.Column(db.DateTime)
    transaction_id = db.Column(db.String(50))

class Batch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(50), unique=True, nullable=False)
    budget = db.Column(db.Float, nullable=False)
    source_filename = db.Column(db.String(255))
    output_filename = db.Column(db.String(255))
    total_applicants = db.Column(db.Integer, default=0)
    total_allocated = db.Column(db.Float, default=0)
    budget_utilization = db.Column(db.Float, default=0)
    fairness_metrics_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ThankYouMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    application_id = db.Column(db.Integer, db.ForeignKey('application.id'))
    title = db.Column(db.String(140), nullable=False)
    message = db.Column(db.Text, nullable=False)
    visibility = db.Column(db.String(20), default='private')
    reviewed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('thank_you_messages', lazy='dynamic'))
    application = db.relationship('Application', backref=db.backref('thank_you_messages', lazy='dynamic'))

class CommunicationMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(160), nullable=False)
    body = db.Column(db.Text, nullable=False)
    audience = db.Column(db.String(30), default='all')
    priority = db.Column(db.String(20), default='normal')
    status = db.Column(db.String(20), default='sent')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sender = db.relationship('User', backref=db.backref('communication_messages', lazy='dynamic'))

class SupportTicket(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), default='General')
    title = db.Column(db.String(160), nullable=False)
    message = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), default='normal')
    status = db.Column(db.String(20), default='open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('support_tickets', lazy='dynamic'))

class UserSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    email_notifications = db.Column(db.Boolean, default=True)
    sms_notifications = db.Column(db.Boolean, default=False)
    compact_tables = db.Column(db.Boolean, default=False)
    default_budget = db.Column(db.Float, default=1000000)
    theme = db.Column(db.String(20), default='modern')
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('settings', uselist=False))

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Custom JSON encoder to handle NaN, infinity, and other types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                logger.debug(f"Replacing NaN/Infinity with 0: {obj}")
                return 0
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as e:
            logger.error(f"JSON serialization error for object {obj}: {str(e)}")
            return str(obj)
app.json_encoder = CustomJSONEncoder

# Serve static files with cache-control headers
@app.route('/static/<path:filename>')
def serve_static(filename):
    response = send_file(os.path.join('static', filename))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({'error': e.description, 'status': 'error'}), e.code
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': str(e), 'status': 'error'}), 500

# Load models
try:
    rf_model = joblib.load('rf_model_tuned.joblib')
    xgb_model = joblib.load('xgb_model_tuned.joblib')
    lgb_model = joblib.load('lgb_model_tuned.joblib')
    meta_model = joblib.load('meta_model_tuned.joblib')
except Exception as e:
    logger.error(f"Error loading models: {str(e)}", exc_info=True)
    raise

# Load PCA
try:
    pca = joblib.load('pca.joblib')
except Exception as e:
    logger.error(f"Error loading PCA: {str(e)}", exc_info=True)
    raise

# Load final_feature_names to ensure correct feature alignment
try:
    final_feature_names = joblib.load('final_feature_names.joblib')
    model_input_features = final_feature_names  # Use the exact features the model was trained on
except Exception as e:
    logger.error(f"Error loading final_feature_names: {str(e)}", exc_info=True)
    # Fallback to hardcoded features if loading fails
    model_input_features = [
        'PC1', 'PC2', 'log_financial_burden', 'remaining_fee_transformed', 
        'log_financial_burden_transformed', 'course_completion_ratio'
    ]

logger.debug(f"Model input features: {model_input_features}")

# Define PCA features (consistent with training)
pca_features = [
    'acad_level_University', 'caregiver_Guardian', 'past_ngcdf_Yes',
    'other_support_Yes', 'pg_disability_No', 'study_mode_Day scholar',
    'study_mode_Government Sponsored', 'study_mode_Self sponsored',
    'vuln_index', 'years_remaining', 'course_completion_ratio',
    'acad_perf_Good', 'past_support_impact', 'log_financial_burden'
]

# Define original features
original_features = [
    'amt_applied', 'vuln_index', 'acad_level_University',
    'caregiver_Guardian', 'past_ngcdf_Yes', 'other_support_Yes',
    'pg_disability_No', 'study_mode_Day scholar',
    'study_mode_Government Sponsored', 'study_mode_Self sponsored',
    'years_remaining', 'course_completion_ratio', 'acad_perf_Good',
    'past_support_impact', 'log_financial_burden'
]

# Define state features for allocation (aligned with the thesis DDPG state contract)
state_features = [
    'predicted_financial_need_score',
    'vulnerability_score',
    'amt_applied',
    'PC1',
    'PC2',
    'PC3',
    'PC4'
]

# Form field mapping
form_field_mapping = {
    'Amount Applied (Kshs)': 'amt_applied',
    'Academic Level': 'acad_level_University',
    'Care of Parent/Guardian': 'caregiver_Guardian',
    'Have you received bursaries from NG-CDF in the past': 'past_ngcdf_Yes',
    'Financial support/bursaries from other organizations': 'other_support_Yes',
    'Parent/Guardian Disability/Chronic Disease Status': 'pg_disability_No',
    'Mode of Study: Day scholar': 'study_mode_Day scholar',
    'Mode of Study: Government Sponsored': 'study_mode_Government Sponsored',
    'Mode of Study: Self sponsored': 'study_mode_Self sponsored',
    'Years Remaining': 'years_remaining',
    'Course Completion Ratio': 'course_completion_ratio',
    'Average Academic Performance': 'acad_perf_Good',
    'Past Support Impact': 'past_support_impact',
    'Log Financial Burden': 'log_financial_burden'
}

# Upload schema contract
UPLOAD_COLUMN_MAPPING = {
    'Amount Applied for (KSh)': 'Amount Applied (Kshs)',
    'Fee Balance (KSh)': 'Fee Balance',
    'Expected Year of Course Completion': 'Year of Course Completion',
    'Kindly indicate your family status.': 'Family status',
    'Family status.': 'Family status',
    'Care of Parent/Guardian': 'Care of',
    'Type of employment': 'Employment Type',
    'Is it a main source of income': 'Is it a main Source of Income()',
    'Have you received bursaries from NG-CDF in the past': 'Past Financial Support(NG-CDF)',
    'Financial support/bursaries from other organizations': 'Past Financial Support(Others)',
    'When you last received the support(Year)': 'Last Received',
    'Parent/Guardian Disability/Chronic Disease Status': 'Parent/Guardian Disability Status',
    'All Supportive documents Available': 'Supportive documents Available'
}

REQUIRED_UPLOAD_COLUMNS = [
    'Name', 'Gender', 'Ward', 'Academic Level', 'Average Academic Performance',
    'School', 'Course of Study', 'Course Duration', 'Mode of Study',
    'Year of Course Completion', 'Amount Applied (Kshs)', 'Fee Balance',
    'Family status', 'Care of', 'Employment Type', 'Past Financial Support(NG-CDF)',
    'Past Financial Support(Others)', 'Individual Disability Status',
    'Parent/Guardian Disability Status', 'Supportive documents Available',
    'Recommendation'
]

NUMERIC_UPLOAD_COLUMNS = [
    'Year of Application', 'Course Duration', 'Year of Course Completion',
    'Amount Applied (Kshs)', 'Fee Balance', 'If yes, specify how much.', 'Last Received'
]

# Global variables
remaining_budget = None
original_budget = None

def ensure_database_schema():
    with app.app_context():
        db.create_all()
        try:
            inspector = inspect(db.engine)
            if 'batch' not in inspector.get_table_names():
                return
            existing_columns = {column['name'] for column in inspector.get_columns('batch')}
            column_defs = {
                'source_filename': 'VARCHAR(255)',
                'output_filename': 'VARCHAR(255)',
                'total_applicants': 'INTEGER DEFAULT 0',
                'total_allocated': 'FLOAT DEFAULT 0',
                'budget_utilization': 'FLOAT DEFAULT 0',
                'fairness_metrics_json': 'TEXT'
            }
            with db.engine.begin() as connection:
                for column, definition in column_defs.items():
                    if column not in existing_columns:
                        connection.exec_driver_sql(f'ALTER TABLE batch ADD COLUMN {column} {definition}')
        except Exception as e:
            logger.warning(f"Could not auto-update database schema: {str(e)}")

ensure_database_schema()

# Function to replace NaN in dictionaries
def replace_nan_in_dict(d):
    if isinstance(d, dict):
        return {k: replace_nan_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nan_in_dict(item) for item in d]
    elif isinstance(d, (np.floating, float)):
        if np.isnan(d) or np.isinf(d):
            logger.debug(f"Replacing NaN/Infinity with 0 in dict: {d}")
            return 0
        return float(d)
    elif isinstance(d, np.ndarray):
        return np.nan_to_num(d, nan=0.0).tolist()
    return d

def parse_number(value, default=0.0):
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        value = value.strip().replace(',', '')
        if value.lower() in ['', 'none', 'nan', 'n/a', 'na']:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def parse_int(value, default=0):
    return int(parse_number(value, default))

def normalize_upload_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.encode('utf-8').str.decode('utf-8').str.strip().str.replace('  ', ' ')
    df.rename(columns=UPLOAD_COLUMN_MAPPING, inplace=True)
    return df

def validate_upload_dataframe(df):
    errors = []
    warnings = []

    missing_columns = [column for column in REQUIRED_UPLOAD_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append({
            'type': 'missing_columns',
            'message': 'The CSV is missing required columns.',
            'columns': missing_columns
        })

    if df.empty:
        errors.append({
            'type': 'empty_file',
            'message': 'The CSV has no applicant rows.'
        })
        return errors, warnings

    if errors:
        return errors, warnings

    duplicate_names = df['Name'].fillna('').str.strip()
    duplicate_count = int(duplicate_names[duplicate_names.duplicated() & duplicate_names.ne('')].count())
    if duplicate_count:
        warnings.append({
            'type': 'duplicate_names',
            'message': f'{duplicate_count} duplicate applicant name rows were found. They will still be processed.'
        })

    for column in NUMERIC_UPLOAD_COLUMNS:
        if column not in df.columns:
            continue
        invalid_rows = []
        for idx, value in df[column].items():
            if column in ['If yes, specify how much.', 'Last Received'] and (pd.isna(value) or str(value).strip() == ''):
                continue
            parsed = parse_number(value, None)
            if parsed is None:
                invalid_rows.append(int(idx) + 2)
        if invalid_rows:
            errors.append({
                'type': 'invalid_numeric',
                'message': f'{column} contains values that cannot be read as numbers.',
                'column': column,
                'rows': invalid_rows[:20],
                'total_rows': len(invalid_rows)
            })

    amount_errors = []
    for idx, value in df['Amount Applied (Kshs)'].items():
        if parse_number(value, 0) <= 0:
            amount_errors.append(int(idx) + 2)
    if amount_errors:
        errors.append({
            'type': 'invalid_amount_applied',
            'message': 'Amount Applied (Kshs) must be greater than zero.',
            'rows': amount_errors[:20],
            'total_rows': len(amount_errors)
        })

    balance_errors = []
    for idx, value in df['Fee Balance'].items():
        if parse_number(value, 0) < 0:
            balance_errors.append(int(idx) + 2)
    if balance_errors:
        errors.append({
            'type': 'invalid_fee_balance',
            'message': 'Fee Balance cannot be negative.',
            'rows': balance_errors[:20],
            'total_rows': len(balance_errors)
        })

    year_warnings = []
    for idx, row in df.iterrows():
        app_year = parse_int(row.get('Year of Application', 2025), 2025)
        completion_year = parse_int(row.get('Year of Course Completion', 2025), 2025)
        if completion_year < app_year:
            year_warnings.append(int(idx) + 2)
    if year_warnings:
        warnings.append({
            'type': 'completion_before_application',
            'message': 'Some completion years are earlier than application years; years remaining will be clamped.',
            'rows': year_warnings[:20],
            'total_rows': len(year_warnings)
        })

    return errors, warnings

def normalize_scores(values, lower=0.01, upper=0.99):
    scores = np.asarray(values, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    min_score = scores.min()
    max_score = scores.max()
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
        return np.clip(scores, lower, upper)
    return np.full(scores.shape, (lower + upper) / 2)

def build_state_data(state_df):
    state_df = state_df.copy()
    for feature in state_features:
        if feature not in state_df.columns:
            state_df[feature] = 0
    state_df = state_df[state_features].apply(pd.to_numeric, errors='coerce')
    state_df = state_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(state_df) > 1 and any(state_df[col].max() > state_df[col].min() for col in state_df.columns):
        state_data = MinMaxScaler().fit_transform(state_df)
    else:
        state_data = state_df.to_numpy(dtype=np.float64)
        if 'predicted_financial_need_score' in state_features:
            idx = state_features.index('predicted_financial_need_score')
            state_data[:, idx] = np.clip(state_data[:, idx], 0, 1)
        if 'vulnerability_score' in state_features:
            idx = state_features.index('vulnerability_score')
            state_data[:, idx] = np.clip(state_data[:, idx] / 10.0, 0, 1)
        if 'amt_applied' in state_features:
            idx = state_features.index('amt_applied')
            state_data[:, idx] = np.clip(state_data[:, idx] / 100000.0, 0, 1)
        for feature in ['PC1', 'PC2', 'PC3', 'PC4']:
            if feature in state_features:
                idx = state_features.index(feature)
                state_data[:, idx] = expit(state_data[:, idx])

    return np.nan_to_num(state_data, nan=0.0, posinf=1.0, neginf=0.0)

def allocation_reason(row, score, allocation):
    reasons = []
    fee_balance = parse_number(row.get('Fee Balance', 0))
    amount_applied = parse_number(row.get('Amount Applied (Kshs)', 0))
    support_amount = parse_number(row.get('If yes, specify how much.', 0))
    family_status = str(row.get('Family status', '')).strip()
    ind_disability = str(row.get('Individual Disability Status', '')).strip()
    pg_disability = str(row.get('Parent/Guardian Disability Status', '')).strip()
    past_ngcdf = str(row.get('Past Financial Support(NG-CDF)', '')).strip()
    documents = str(row.get('Supportive documents Available', '')).strip()
    recommendation = str(row.get('Recommendation', '')).strip()

    if score >= 0.75:
        reasons.append('High predicted financial need')
    elif score >= 0.5:
        reasons.append('Moderate predicted financial need')
    else:
        reasons.append('Lower relative need score')
    if fee_balance > 0 and amount_applied > 0 and fee_balance >= amount_applied:
        reasons.append('Fee balance meets or exceeds requested support')
    if support_amount <= 0 and past_ngcdf != 'Yes':
        reasons.append('No recent recorded bursary support')
    if family_status in ['Single Parent', 'Partial orphan', 'Total Orphan']:
        reasons.append(f'Family vulnerability: {family_status}')
    if ind_disability == 'Yes' or pg_disability in ['Yes', 'Chronic Disease', 'Both']:
        reasons.append('Disability or chronic illness factor')
    if documents in ['Yes', 'Partial']:
        reasons.append('Supportive documents available')
    if recommendation == 'Decline':
        reasons.append('Recommendation marked decline')
    if allocation <= 0:
        reasons.append('No allocation after budget constraints')
    return '; '.join(reasons[:5])

def allocation_reason_for_application(application, score, allocation):
    row = {
        'Fee Balance': application.fee_balance,
        'Amount Applied (Kshs)': application.amount_applied,
        'If yes, specify how much.': application.other_support_amount,
        'Family status': application.family_status,
        'Individual Disability Status': application.individual_disability,
        'Parent/Guardian Disability Status': application.parent_disability,
        'Past Financial Support(NG-CDF)': application.past_ngcdf_support,
        'Supportive documents Available': application.documents_available,
        'Recommendation': application.recommendation
    }
    return allocation_reason(row, score, allocation)

def resolve_output_file(filename):
    safe_name = os.path.basename(filename or '')
    if not safe_name.startswith('allocation_output_') or not safe_name.endswith('.xlsx'):
        raise ValueError('Invalid allocation output filename.')
    path = os.path.abspath(os.path.join(app.root_path, safe_name))
    if not path.startswith(os.path.abspath(app.root_path)):
        raise ValueError('Invalid file path.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{safe_name} was not found.')
    return path

# Actor Network for DDPG
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        return self.max_action * x

# Bursary Allocation Environment
class BursaryAllocationEnv:
    def __init__(self, state_space_np, amt_applied_np, need_scores_np, budget):
        self.state_space_np = state_space_np
        self.amt_applied_np = amt_applied_np
        self.need_scores_np = need_scores_np
        self.budget = budget
        self.current_idx = 0
        self.remaining_budget = budget
        self.allocations = np.zeros(len(state_space_np), dtype=np.float64)
        self.min_allocation = 10.0
        self.state_dim = state_space_np.shape[1]
        self.action_dim = 1
        self.total_students = len(state_space_np)
        self.max_applied = np.max(amt_applied_np)
        self.high_need_students = np.sum(need_scores_np > 0.5)
        self.non_high_need_students = np.sum(need_scores_np <= 0.5)
        self.indices = np.arange(self.total_students)
        self.allocated_students = 0
        self.high_need_allocated = 0
        self.non_high_need_allocated = 0
        self.budget_to_applied_ratio = budget / np.sum(amt_applied_np) if np.sum(amt_applied_np) > 0 else 1.0

    def reset(self):
        self.current_idx = 0
        self.remaining_budget = self.budget
        self.allocations = np.zeros(len(self.state_space_np), dtype=np.float64)
        self.allocated_students = 0
        self.high_need_allocated = 0
        self.non_high_need_allocated = 0
        np.random.shuffle(self.indices)
        return self.state_space_np[self.indices[self.current_idx]]

    def step(self, action):
        idx = self.indices[self.current_idx]
        need_score = self.need_scores_np[idx]
        applied_amount = self.amt_applied_np[idx]
        already_allocated = self.allocations[idx]
        remaining_applied = max(0, applied_amount - already_allocated)

        if self.remaining_budget <= 0:
            allocation = 0
            done = True
        else:
            policy_action = float(np.ravel(action)[0]) if action is not None else 1.0
            policy_action = np.clip(policy_action, 0, 1)
            adjusted_percentage = self.budget_to_applied_ratio * need_score * policy_action
            adjusted_percentage = np.clip(adjusted_percentage, 0, 1)
            allocation = remaining_applied * adjusted_percentage
            if allocation > 0:
                allocation = max(allocation, self.min_allocation)
            allocation = min(allocation, remaining_applied)
            if allocation > self.remaining_budget:
                allocation = min(allocation, self.remaining_budget, remaining_applied, self.min_allocation)
                done = False if self.remaining_budget > 0 else True
            else:
                done = False

        if allocation > 0 and self.allocations[idx] == 0:
            self.allocated_students += 1
            if need_score > 0.5:
                self.high_need_allocated += 1
            else:
                self.non_high_need_allocated += 1

        self.remaining_budget -= allocation
        self.allocations[idx] += allocation

        self.current_idx += 1
        done = done or self.remaining_budget <= 0 or self.current_idx >= len(self.state_space_np)

        next_state = self.state_space_np[self.indices[self.current_idx]] if not done else np.zeros(self.state_dim)
        return next_state, 0, done, {}

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim, max_action)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state).detach().numpy()
        need_score = state[0].item()
        if need_score > 0.5:
            action = np.clip(action, 0.3, 1)
        else:
            action = np.clip(action, 0.2, 1)
        return np.clip(action, 0, 1)

# Root route
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role in ['admin', 'stakeholder']:
            return render_template('index.html')
        elif current_user.role == 'applicant':
            return redirect(url_for('applications'))
    return redirect(url_for('login'))

# Register endpoint
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        role = request.form.get('role', 'applicant')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please use a different email or log in.', 'danger')
            return redirect(url_for('register'))

        user = User(email=email, name=name, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login endpoint
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            elif user.role == 'stakeholder':
                return redirect(url_for('index'))
            elif user.role == 'applicant':
                return redirect(url_for('applications'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout endpoint
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# Dashboard page (protected for admins)
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        if current_user.role != 'admin':
            flash('You do not have permission to access the dashboard.', 'danger')
            return redirect(url_for('index'))

        allocation_count = Allocation.query.count()
        logger.debug(f"Allocations found: {allocation_count}")
        if allocation_count == 0:
            logger.warning("No allocation data available for dashboard")
            return render_template('dashboard.html', data_available=False, message="No allocation data available. Please perform a batch allocation first.")
        return render_template('dashboard.html', data_available=True)
    except Exception as e:
        logger.error(f"Error in dashboard endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Allocation data endpoint
@app.route('/allocation_data', methods=['GET'])
@login_required
def allocation_data():
    try:
        if current_user.role != 'admin':
            return jsonify({'error': 'Unauthorized access.', 'status': 'error'}), 403

        # Find the latest batch
        latest_batch = Batch.query.order_by(Batch.created_at.desc()).first()
        if not latest_batch:
            logger.error("No batch data available in allocation_data endpoint")
            return jsonify({'error': 'No batch data available. Please perform a batch allocation first.', 'status': 'error'}), 404

        batch_id = latest_batch.id
        batch_budget = latest_batch.budget  # Get the budget from the Batch record
        logger.debug(f"Latest Batch ID: {batch_id}, Batch Budget: {batch_budget}")

        allocations = Allocation.query.filter_by(batch_id=batch_id).all()
        if not allocations:
            logger.error(f"No allocation data available for batch ID {batch_id} in allocation_data endpoint")
            return jsonify({'error': 'No allocation data available for the latest batch. Please perform a batch allocation first.', 'status': 'error'}), 404

        logger.debug(f"Retrieved {len(allocations)} allocations for batch ID {batch_id}")

        data = []
        total_allocated = 0
        allocation_data = []
        for allocation in allocations:
            application = Application.query.get(allocation.application_id)
            user = User.query.get(application.user_id)
            row = {
                'Name': application.name or (user.name if user else ''),
                'Gender': user.gender if user else '',
                'Ward': user.ward if user else '',
                'Academic Level': application.academic_level,
                'Year of Course Completion': application.expected_year_of_completion,
                'School Fee Balance': application.fee_balance,
                'Applied Amount': application.amount_applied,
                'Financial Need Score (%)': allocation.financial_need_score * 100 if allocation.financial_need_score else 0,
                'Allocated Amount': allocation.allocated_amount,
                'Allocation Reasons': allocation_reason_for_application(
                    application,
                    allocation.financial_need_score or 0,
                    allocation.allocated_amount or 0
                )
            }
            total_allocated += allocation.allocated_amount if allocation.allocated_amount else 0
            data.append(row)
            allocation_data.append({
                'financial_need_score': allocation.financial_need_score * 100 if allocation.financial_need_score else 0,
                'allocated_amount': allocation.allocated_amount if allocation.allocated_amount else 0
            })

        budget = batch_budget  # Use the budget from the Batch record
        budget_utilization = (total_allocated / budget) * 100 if budget > 0 else 0
        remaining = budget - total_allocated
        logger.debug(f"Dashboard - Total Allocated: {total_allocated}, Budget: {budget}, Utilization: {budget_utilization:.2f}%")

        stats = {
            'total_allocated': float(total_allocated),
            'budget_utilization': float(budget_utilization),
            'remaining_budget': float(remaining)
        }

        fairness_metrics_by_score = {}
        utilization_by_score = {}

        df = pd.DataFrame(allocation_data)
        if not df.empty:
            bins = [0, 29.433, 45.608, 61.783, 77.958, 94.132, 100]
            labels = [
                '(13.178, 29.433]', '(29.433, 45.608]', '(45.608, 61.783]',
                '(61.783, 77.958]', '(77.958, 94.132]', '(94.132, 100]'
            ]
            df['score_bin'] = pd.cut(df['financial_need_score'], bins=bins, labels=labels, include_lowest=True)

            grouped = df.groupby('score_bin', observed=True)
            mean_allocations = grouped['allocated_amount'].mean().fillna(0).round(2)
            counts = grouped['allocated_amount'].count()
            totals = grouped['allocated_amount'].sum().fillna(0).round(2)

            total_allocated_sum = totals.sum()
            utilization_percentages = (totals / total_allocated_sum * 100).fillna(0).round(2)

            for bin_label in labels:
                fairness_metrics_by_score[bin_label] = {
                    'mean': float(mean_allocations.get(bin_label, 0)),
                    'count': int(counts.get(bin_label, 0)),
                    'total_allocated': float(totals.get(bin_label, 0))
                }
                utilization_by_score[bin_label] = float(utilization_percentages.get(bin_label, 0))

        logger.info(f"Allocation Data: {data[:2]}")
        logger.debug(f"Stats: {stats}")
        logger.debug(f"Fairness Metrics by Score: {fairness_metrics_by_score}")
        logger.debug(f"Utilization by Score: {utilization_by_score}")

        response = {
            'data': replace_nan_in_dict(data),
            'stats': replace_nan_in_dict(stats),
            'fairness_metrics_by_score': replace_nan_in_dict(fairness_metrics_by_score),
            'utilization_by_score': replace_nan_in_dict(utilization_by_score),
            'status': 'success'
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in allocation_data endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to fetch allocation data: {str(e)}', 'status': 'error'}), 500

@app.route('/batch_allocation', methods=['GET', 'POST'])
@login_required
def batch_allocation():
    if current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    global remaining_budget, original_budget
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or not file.filename:
                return jsonify({'error': 'Please choose a CSV file to upload.', 'status': 'error'}), 400
            source_filename = secure_filename(file.filename)
            budget = parse_number(request.form.get('budget'), 0)
            logger.debug(f"Received budget: {budget}")

            if budget <= 0:
                logger.error("Budget cannot be zero")
                return jsonify({'error': 'Budget must be greater than zero', 'status': 'error'}), 400

            original_budget = budget

            logger.debug("Reading CSV file")
            input_df = pd.read_csv(file, encoding='utf-8-sig')
            logger.debug(f"Raw column names after loading CSV: {list(input_df.columns)}")

            logger.debug("Normalizing column names")
            input_df = normalize_upload_dataframe(input_df)
            logger.debug(f"Normalized column names: {list(input_df.columns)}")

            validation_errors, validation_warnings = validate_upload_dataframe(input_df)
            if validation_errors:
                return jsonify({
                    'error': 'CSV validation failed. Please fix the listed issues and upload again.',
                    'validation_errors': validation_errors,
                    'validation_warnings': validation_warnings,
                    'status': 'error'
                }), 400
            raw_input_df = input_df.copy()

            logger.debug("Pre-generating hashed password")
            placeholder_password = bcrypt.generate_password_hash("placeholder").decode('utf-8')

            logger.debug("Fetching existing emails")
            existing_emails = set(user.email for user in User.query.with_entities(User.email).all())

            logger.debug("Saving users and applications to database")
            timestamp = int(time.time())
            batch_size = 100
            users_to_add = []
            applications_to_add = []

            for idx, row in input_df.iterrows():
                email = f"applicant_{idx}_{timestamp}@example.com"
                if email not in existing_emails:
                    logger.debug(f"Creating user with email: {email}")
                    user = User(
                        email=email,
                        password=placeholder_password,
                        name=row.get('Name', ''),
                        gender=row.get('Gender', ''),
                        ward=row.get('Ward', ''),
                        role='applicant'
                    )
                    users_to_add.append(user)
                    existing_emails.add(email)

                last_received = row.get('Last Received', 0)
                if pd.isna(last_received):
                    last_received = 0

                logger.debug(f"Preparing application for user email: {email}")
                application = Application(
                    user_id=None,
                    name=row.get('Name', ''),
                    academic_level=row.get('Academic Level', ''),
                    year_of_application=parse_int(row.get('Year of Application', 2025), 2025),
                    average_academic_performance=row.get('Average Academic Performance', ''),
                    school=row.get('School', ''),
                    course_of_study=row.get('Course of Study', ''),
                    course_duration=parse_int(row.get('Course Duration', 0)),
                    mode_of_study=row.get('Mode of Study', ''),
                    expected_year_of_completion=parse_int(row.get('Year of Course Completion', 2025), 2025),
                    amount_applied=parse_number(row.get('Amount Applied (Kshs)', 0)),
                    fee_balance=parse_number(row.get('Fee Balance', 0)),
                    family_status=row.get('Family status', ''),
                    care_of=row.get('Care of', ''),
                    employment_type=row.get('Employment Type', ''),
                    is_main_source_of_income=row.get('Is it a main Source of Income()', ''),
                    past_ngcdf_support=row.get('Past Financial Support(NG-CDF)', ''),
                    other_support=row.get('Past Financial Support(Others)', ''),
                    other_support_amount=parse_number(row.get('If yes, specify how much.', 0)),
                    last_support_year=parse_int(last_received),
                    individual_disability=row.get('Individual Disability Status', ''),
                    parent_disability=row.get('Parent/Guardian Disability Status', ''),
                    documents_available=row.get('Supportive documents Available', ''),
                    recommendation=row.get('Recommendation', '')
                )
                applications_to_add.append((application, email))

                if len(users_to_add) >= batch_size:
                    logger.debug(f"Committing batch of {len(users_to_add)} users")
                    db.session.add_all(users_to_add)
                    db.session.commit()
                    users_to_add = []

            if users_to_add:
                logger.debug(f"Committing final batch of {len(users_to_add)} users")
                db.session.add_all(users_to_add)
                db.session.commit()

            logger.debug("Assigning user IDs to applications")
            created_application_ids = []
            for idx, (application, email) in enumerate(applications_to_add):
                user = User.query.filter_by(email=email).first()
                if user:
                    application.user_id = user.id
                else:
                    logger.error(f"User with email {email} not found after commit")
                    raise Exception(f"User with email {email} not found")

                if (idx + 1) % batch_size == 0 or idx == len(applications_to_add) - 1:
                    logger.debug(f"Committing batch of applications up to index {idx}")
                    db.session.add_all([app for app, _ in applications_to_add[max(0, idx - batch_size + 1):idx + 1]])
                    db.session.commit()

            created_application_ids = [application.id for application, _ in applications_to_add]

            logger.debug("Mapping input data for allocation")
            mapped_data_list = []
            for idx, row in input_df.iterrows():
                mapped_data = {}
                year_of_application = parse_int(row.get('Year of Application', 2025), 2025)
                academic_level = row.get('Academic Level', 'Secondary School')
                mapped_data['acad_level_University'] = 1 if academic_level == 'University' else 0
                acad_perf = row.get('Average Academic Performance', 'Poor')
                mapped_data['acad_perf_Good'] = 1 if acad_perf == 'Good' else 0
                mapped_data['acad_perf_Fair'] = 1 if acad_perf == 'Fair' else 0
                mapped_data['acad_perf_Poor'] = 1 if acad_perf == 'Poor' else 0
                course_duration = parse_number(row.get('Course Duration', 0))
                mapped_data['course_dur'] = course_duration
                mode_of_study = row.get('Mode of Study', 'Boarding')
                mapped_data['study_mode_Boarding'] = 1 if mode_of_study in ['Boarding', 'Bording'] else 0
                mapped_data['study_mode_Day scholar'] = 1 if mode_of_study == 'Day scholar' else 0
                mapped_data['study_mode_Government Sponsored'] = 1 if mode_of_study == 'Government Sponsored' else 0
                mapped_data['study_mode_Self sponsored'] = 1 if mode_of_study == 'Self sponsored' else 0
                expected_completion_year = parse_int(row.get('Year of Course Completion', 2025), 2025)
                mapped_data['exp_completion'] = expected_completion_year
                mapped_data['amt_applied'] = parse_number(row.get('Amount Applied (Kshs)', 0))
                mapped_data['fee_balance'] = parse_number(row.get('Fee Balance', 0))
                family_status = row.get('Family status', 'Other')
                mapped_data['family_status_Total Orphan'] = 1 if family_status == 'Total Orphan' else 0
                mapped_data['family_status_Partial orphan'] = 1 if family_status == 'Partial orphan' else 0
                mapped_data['family_status_Single Parent'] = 1 if family_status == 'Single Parent' else 0
                caregiver = row.get('Care of', 'Mother')
                mapped_data['caregiver_Guardian'] = 1 if caregiver == 'Guardian' else 0
                mapped_data['caregiver_Father'] = 1 if caregiver == 'Father' else 0
                mapped_data['caregiver_Mother'] = 1 if caregiver == 'Mother' else 0
                emp_type = row.get('Employment Type', 'Unknown')
                mapped_data['emp_type_Contractual'] = 1 if emp_type == 'Contractual' else 0
                mapped_data['emp_type_Parmanent'] = 1 if emp_type == 'Permanent' else 0
                mapped_data['emp_type_Retired'] = 1 if emp_type == 'Retired' else 0
                mapped_data['emp_type_Self Employed'] = 1 if emp_type == 'Self Employed' else 0
                mapped_data['emp_type_Unknown'] = 1 if emp_type == 'Unknown' else 0
                mapped_data['is_main_source'] = row.get('Is it a main Source of Income()', 'Yes')
                past_ngcdf = row.get('Past Financial Support(NG-CDF)', 'No')
                mapped_data['past_ngcdf_Yes'] = 1 if past_ngcdf == 'Yes' else 0
                other_support = row.get('Past Financial Support(Others)', 'No')
                mapped_data['other_support_Yes'] = 1 if other_support == 'Yes' else 0
                support_amt = parse_number(row.get('If yes, specify how much.', 0))
                mapped_data['support_amt'] = support_amt
                last_support_year = row.get('Last Received', 0)
                if pd.isna(last_support_year):
                    last_support_year = 0
                current_year = 2025
                past_support_impact = max(0, 1 - (current_year - last_support_year) / 10) if last_support_year > 0 else 0
                mapped_data['past_support_impact'] = past_support_impact
                ind_disability = row.get('Individual Disability Status', 'No')
                mapped_data['ind_disability_Yes'] = 1 if ind_disability == 'Yes' else 0
                pg_disability = row.get('Parent/Guardian Disability Status', 'No')
                mapped_data['pg_disability_No'] = 1 if pg_disability == 'No' else 0
                mapped_data['pg_disability_Yes'] = 1 if pg_disability == 'Yes' else 0
                mapped_data['pg_disability_Chronic Disease'] = 1 if pg_disability == 'Chronic Disease' else 0
                mapped_data['pg_disability_Both'] = 1 if pg_disability == 'Both' else 0
                docs_available = row.get('Supportive documents Available', 'No')
                mapped_data['docs_available_No'] = 1 if docs_available == 'No' else 0
                mapped_data['docs_available_Partial'] = 1 if docs_available == 'Partial' else 0
                mapped_data['docs_available_Yes'] = 1 if docs_available == 'Yes' else 0
                recommendation = row.get('Recommendation', 'Approve')
                mapped_data['recommendation_Decline'] = 1 if recommendation == 'Decline' else 0
                vuln_index = (
                    mapped_data['ind_disability_Yes'] +
                    mapped_data['pg_disability_Yes'] +
                    mapped_data['pg_disability_Chronic Disease'] +
                    mapped_data['pg_disability_Both'] +
                    mapped_data['family_status_Total Orphan'] +
                    mapped_data['family_status_Partial orphan'] +
                    mapped_data['family_status_Single Parent']
                ) / 7.0
                mapped_data['vuln_index'] = vuln_index
                years_remaining = max(0, expected_completion_year - year_of_application)
                mapped_data['years_remaining'] = years_remaining
                course_completion_ratio = (course_duration - years_remaining) / course_duration if course_duration > 0 else 0
                mapped_data['course_completion_ratio'] = max(0, min(1, course_completion_ratio))
                fee_balance = mapped_data['fee_balance']
                log_financial_burden = np.log1p(fee_balance) / np.log1p(1000000) if fee_balance > 0 else 0
                mapped_data['log_financial_burden'] = log_financial_burden
                # Add remaining_fee for transformation (fix: use max() instead of clip() for scalar float)
                remaining_fee = max(0, fee_balance - support_amt)
                mapped_data['remaining_fee'] = remaining_fee
                mapped_data_list.append(mapped_data)

            logger.debug("Converting mapped data to DataFrame")
            input_df = pd.DataFrame(mapped_data_list)
            for feature in original_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            logger.debug(f"Processed DataFrame head:\n{input_df.head().to_string()}")

            logger.debug("Logging feature variation")
            logger.debug(f"Feature variation (std dev):\n{input_df[original_features].std().to_string()}")
            logger.debug(f"Feature min values:\n{input_df[original_features].min().to_string()}")
            logger.debug(f"Feature max values:\n{input_df[original_features].max().to_string()}")

            # Apply Yeo-Johnson transformation to features
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            features_for_score = ['remaining_fee', 'vulnerability_score', 'log_financial_burden', 
                                 'course_completion_ratio', 'past_support_impact']
            input_df['vulnerability_score'] = (
                3 * input_df.get('ind_disability_Yes', 0) +
                2 * input_df.get('pg_disability_Yes', 0) +
                2 * input_df.get('pg_disability_Chronic Disease', 0) +
                4 * input_df.get('pg_disability_Both', 0) +
                3 * input_df.get('family_status_Total Orphan', 0) +
                2 * input_df.get('family_status_Partial orphan', 0) +
                1 * input_df.get('family_status_Single Parent', 0) +
                1 * (1 - input_df['past_ngcdf_Yes']) +
                np.log1p(input_df['remaining_fee']) / np.log1p(input_df['remaining_fee'].max()) * 5 +
                (input_df['course_completion_ratio'] / input_df['course_completion_ratio'].max()) * 2 +
                (1 - input_df['course_dur'] / 4) * 2 +
                1 * input_df.get('docs_available_No', 0) +
                2 * input_df.get('emp_type_Contractual', 0) +
                1 * input_df.get('emp_type_Unknown', 0) +
                1 * input_df.get('study_mode_Boarding', 0) +
                2 * input_df.get('acad_level_University', 0)
            ).fillna(0)
            max_vuln = input_df['vulnerability_score'].max()
            if max_vuln > 0:
                input_df['vulnerability_score'] = (input_df['vulnerability_score'] / max_vuln) * 10

            for feature in ['remaining_fee', 'vulnerability_score', 'log_financial_burden']:
                input_df[f'{feature}_transformed'] = pt.fit_transform(input_df[[feature]])
            features_for_score = [f'{feature}_transformed' if feature in ['remaining_fee', 'vulnerability_score', 'log_financial_burden'] else feature for feature in features_for_score]

            # Standardize features
            scaler_std = StandardScaler()
            input_df[features_for_score] = scaler_std.fit_transform(input_df[features_for_score])

            # Learn weights for financial need score
            X_score = input_df[features_for_score].fillna(0)
            y_score = input_df['amt_applied'].fillna(input_df['amt_applied'].median())
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_score, y_score)
            learned_weights = pd.Series(ridge_model.coef_, index=features_for_score).to_dict()
            logger.debug(f"Learned Financial Need Score Weights: {learned_weights}")

            # Calculate financial need score
            input_df['financial_need_score'] = (
                sum(learned_weights[feature] * input_df[feature] for feature in features_for_score) +
                0.1 * (input_df['remaining_fee_transformed'] * input_df['vulnerability_score_transformed'])
            )

            # Standardize before sigmoid transformation
            input_df['financial_need_score'] = (input_df['financial_need_score'] - input_df['financial_need_score'].mean()) / input_df['financial_need_score'].std()

            # Apply sigmoid transformation
            input_df['financial_need_score'] = expit(input_df['financial_need_score'])

            # Scale to 0-100 with clipping, handling identical scores
            min_score = input_df['financial_need_score'].min()
            max_score = input_df['financial_need_score'].max()
            if max_score > min_score:
                input_df['financial_need_score'] = 100 * (input_df['financial_need_score'] - min_score) / (max_score - min_score)
            else:
                logger.warning("All financial need scores are identical after transformation; setting to 0.")
                input_df['financial_need_score'] = 0
            input_df['financial_need_score'] = input_df['financial_need_score'].clip(lower=1, upper=99)

            logger.debug("Performing PCA transformation")
            pca_input_df = input_df[pca_features]
            X_pca = pca.transform(pca_input_df)
            X_pca = np.nan_to_num(X_pca, nan=0.0)
            logger.debug(f"PCA output (first 5 rows):\n{X_pca[:5]}")
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])

            # Include transformed features in input_df_pca
            input_df_pca = pd.concat([input_df[['amt_applied', 'vulnerability_score', 'course_completion_ratio', 'acad_perf_Good', 'past_support_impact', 'log_financial_burden', 'remaining_fee', 'remaining_fee_transformed', 'log_financial_burden_transformed']], pca_df], axis=1)
            logger.debug(f"Columns in input_df_pca before adding predicted_financial_need_score: {list(input_df_pca.columns)}")

            # Scale the input features to the meta-model
            scaler_meta = MinMaxScaler()
            model_input = scaler_meta.fit_transform(input_df_pca[model_input_features])
            model_input_df = pd.DataFrame(model_input, columns=model_input_features)
            logger.debug(f"Scaled model input (first 5 rows):\n{model_input_df.head().to_string()}")

            # Debug: Log the number of features expected by the model and provided
            logger.debug(f"Number of features expected by RandomForestRegressor: {rf_model.n_features_in_}")
            logger.debug(f"Number of features in model_input: {model_input.shape[1]}")
            logger.debug(f"Features provided: {model_input_features}")

            # Predict financial need scores
            logger.debug("Predicting financial need scores")
            rf_pred = np.nan_to_num(rf_model.predict(model_input), nan=0.0)
            xgb_pred = np.nan_to_num(xgb_model.predict(model_input), nan=0.0)
            lgb_pred = np.nan_to_num(lgb_model.predict(model_input), nan=0.0)
            logger.debug(f"RF predictions (first 5): {rf_pred[:5]}")
            logger.debug(f"XGB predictions (first 5): {xgb_pred[:5]}")
            logger.debug(f"LGB predictions (first 5): {lgb_pred[:5]}")
            meta_features = np.column_stack((rf_pred, xgb_pred, lgb_pred))
            financial_need_scores = meta_model.predict(meta_features)
            logger.debug(f"Meta model raw predictions (first 5): {financial_need_scores[:5]}")

            financial_need_scores = np.nan_to_num(financial_need_scores, nan=0.0)
            financial_need_scores = np.clip(financial_need_scores, 0, 1)
            logger.debug(f"After clipping (first 5): {financial_need_scores[:5]}")

            model_score_std = float(np.std(financial_need_scores))
            model_cap_ratio = float(np.mean(financial_need_scores >= 0.98))
            if model_score_std < 0.001 or model_cap_ratio > 0.5:
                logger.warning(
                    "Meta-model scores are saturated; using engineered financial_need_score distribution instead. "
                    f"std={model_score_std:.6f}, cap_ratio={model_cap_ratio:.2f}"
                )
                financial_need_scores = normalize_scores(input_df['financial_need_score'].values / 100.0)
            else:
                financial_need_scores = normalize_scores(expit(StandardScaler().fit_transform(financial_need_scores.reshape(-1, 1)).flatten()))

            # Safe logging
            if np.isscalar(financial_need_scores):
                logger.debug(f"Meta model financial need scores is a scalar: {financial_need_scores}")
            else:
                logger.debug(f"Meta model financial need scores (first 5): {financial_need_scores[:5]}")
            logger.debug(f"Financial need scores variation (std dev): {np.std(financial_need_scores)}")

            logger.debug("Preparing data for allocation")
            input_df_pca['predicted_financial_need_score'] = financial_need_scores
            logger.debug(f"Columns in input_df_pca after adding predicted_financial_need_score: {list(input_df_pca.columns)}")

            missing_features = [feature for feature in state_features if feature not in input_df_pca.columns]
            if missing_features:
                logger.error(f"Missing features in input_df_pca: {missing_features}")
                raise ValueError(f"Missing features in input_df_pca: {missing_features}")

            state_df = input_df_pca[state_features]
            logger.debug(f"Columns in state_df before scaling: {list(state_df.columns)}")
            logger.debug(f"state_df shape: {state_df.shape}")
            logger.debug(f"state_df head:\n{state_df.head().to_string()}")
            if state_df.empty:
                logger.error("state_df is empty")
                raise ValueError("state_df is empty")
            if not all(state_df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.error(f"Non-numeric columns in state_df: {state_df.dtypes}")
                raise ValueError("All columns in state_df must be numeric")
            states = build_state_data(state_df)
            logger.debug(f"Type of states: {type(states)}")
            logger.debug(f"Shape of states: {states.shape}")
            logger.debug(f"First few rows of states:\n{states[:5]}")

            if states.shape[1] != len(state_features):
                logger.error(f"State data has {states.shape[1]} features, but expects {len(state_features)} features")
                raise ValueError(f"State data has {states.shape[1]} features, but expects {len(state_features)} features")

            # Prepare for DDPG allocation
            amt_applied_np = input_df_pca['amt_applied'].values
            need_scores_np = financial_need_scores
            total_applied_amount = np.sum(amt_applied_np)
            budget_to_applied_ratio = budget / total_applied_amount if total_applied_amount > 0 else 1.0
            logger.debug(f"Total Applied Amount: {total_applied_amount} KSh, Budget: {budget} KSh, Budget-to-Applied Ratio: {budget_to_applied_ratio:.4f}")

            # Initialize DDPG environment and agent
            env = BursaryAllocationEnv(states, amt_applied_np, need_scores_np, budget)
            logger.debug(f"Type of env.state_space_np: {type(env.state_space_np)}")
            logger.debug(f"Shape of env.state_space_np: {env.state_space_np.shape}")
            agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim, max_action=1.0)

            # Load the trained DDPG model
            try:
                agent.actor.load_state_dict(torch.load('ddpg_actor.pth', map_location=torch.device('cpu')))
                logger.debug("Successfully loaded DDPG actor model from 'ddpg_actor.pth'")
            except Exception as e:
                logger.error(f"Failed to load DDPG actor model: {str(e)}")
                raise Exception(f"Failed to load DDPG actor model: {str(e)}")

            # Allocate using the DDPG agent
            env.reset()
            allocations = []
            indices = list(range(len(env.state_space_np)))

            agent.actor.eval()
            with torch.no_grad():
                for idx in indices:
                    state = env.state_space_np[idx]
                    applied_amount = env.amt_applied_np[idx]
                    already_allocated = env.allocations[idx]
                    remaining_applied = max(0, applied_amount - already_allocated)
                    need_score = env.need_scores_np[idx]
                    policy_action = float(np.ravel(agent.select_action(state))[0])
                    adjusted_percentage = env.budget_to_applied_ratio * need_score * policy_action
                    adjusted_percentage = np.clip(adjusted_percentage, 0, 1)
                    allocation = remaining_applied * adjusted_percentage
                    if allocation > 0:
                        allocation = max(allocation, env.min_allocation)
                    allocation = min(allocation, remaining_applied)
                    if allocation > env.remaining_budget:
                        allocation = min(allocation, env.remaining_budget, remaining_applied, env.min_allocation)
                    env.remaining_budget -= allocation
                    env.allocations[idx] += allocation
                    allocations.append(allocation)
                    if env.remaining_budget <= 0:
                        break

                # Post-allocation adjustment: Redistribute remaining budget proportionally
                if env.remaining_budget > 0:
                    initial_total_allocated = np.sum(env.allocations)
                    if initial_total_allocated > 0:
                        # Scale up allocations proportionally to use the full budget
                        scaling_factor = env.budget / initial_total_allocated
                        env.allocations *= scaling_factor
                        allocations = env.allocations.tolist()
                        # Cap at applied amounts
                        for i in range(len(allocations)):
                            allocations[i] = min(allocations[i], env.amt_applied_np[i])
                        env.allocations = np.array(allocations)
                        # Recalculate total allocated after capping
                        total_allocated_after_scaling = np.sum(env.allocations)
                        env.remaining_budget = env.budget - total_allocated_after_scaling

                        # Distribute any remaining budget to high-need students
                        if env.remaining_budget > 0:
                            high_need_indices = [i for i in range(len(env.state_space_np)) if env.need_scores_np[i] > 0.5 and env.allocations[i] < env.amt_applied_np[i]]
                            if high_need_indices:
                                high_need_sorted = sorted(high_need_indices, key=lambda i: env.need_scores_np[i], reverse=True)
                                remaining_to_distribute = env.remaining_budget
                                for i in high_need_sorted:
                                    if remaining_to_distribute <= 0:
                                        break
                                    remaining_applied = max(0, env.amt_applied_np[i] - env.allocations[i])
                                    additional_allocation = min(remaining_to_distribute, remaining_applied)
                                    env.allocations[i] += additional_allocation
                                    allocations[i] += additional_allocation
                                    remaining_to_distribute -= additional_allocation
                                env.remaining_budget = remaining_to_distribute

            # Update global remaining_budget
            allocations = env.allocations.tolist()
            remaining_budget = env.remaining_budget
            logger.debug(f"Updated remaining_budget: {remaining_budget}")

            # Save allocations to database
            batch_id_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch = Batch(batch_id=batch_id_str, budget=budget, source_filename=source_filename)
            db.session.add(batch)
            db.session.commit()
            batch_id = batch.id
            logger.debug(f"Created Batch with ID {batch_id}, batch_id_str: {batch_id_str}, budget: {budget}")

            # Retrieve the applications created from this upload, preserving CSV row order.
            applications_by_id = {
                application.id: application
                for application in Application.query.filter(Application.id.in_(created_application_ids)).all()
            }
            applications = [applications_by_id.get(application_id) for application_id in created_application_ids]
            if len(applications) != len(states):
                logger.error(f"Mismatch between number of applications ({len(applications)}) and states ({len(states)})")
                raise ValueError(f"Mismatch between number of applications ({len(applications)}) and states ({len(states)})")
            if any(application is None for application in applications):
                missing_ids = [application_id for application_id, application in zip(created_application_ids, applications) if application is None]
                logger.error(f"Applications missing after upload commit: {missing_ids[:10]}")
                raise ValueError("Some uploaded applications could not be retrieved after saving.")
            logger.debug(f"Retrieved {len(applications)} applications for allocation")

            total_students = len(states)
            for idx in range(total_students):
                application = applications[idx]
                allocation = allocations[idx]
                allocation_record = Allocation(
                    application_id=application.id,
                    batch_id=batch_id,
                    financial_need_score=financial_need_scores[idx],
                    allocated_amount=allocation
                )
                db.session.add(allocation_record)
                application.status = 'approved' if allocation > 0 else 'declined'

            db.session.commit()

            # Verify the number of allocations saved
            saved_allocations = Allocation.query.filter_by(batch_id=batch_id).all()
            logger.debug(f"Number of allocations saved in batch {batch_id}: {len(saved_allocations)}")

            logger.debug("Preparing output DataFrame")
            output_df = pd.DataFrame()
            output_df['Name'] = [app.name or (User.query.get(app.user_id).name if User.query.get(app.user_id) else '') for app in applications]
            output_df['Gender'] = [User.query.get(app.user_id).gender if User.query.get(app.user_id) else '' for app in applications]
            output_df['Ward'] = [User.query.get(app.user_id).ward if User.query.get(app.user_id) else '' for app in applications]
            output_df['Academic Level'] = [app.academic_level for app in applications]
            output_df['Expected Year of Course Completion'] = [app.expected_year_of_completion for app in applications]
            output_df['Fee Balance'] = [app.fee_balance for app in applications]
            output_df['Amount Applied (Kshs)'] = [app.amount_applied for app in applications]
            output_df['Financial Need Score (%)'] = [(score * 100).round(2) for score in financial_need_scores]
            output_df['Allocated Amount'] = allocations
            output_df['Allocation Reasons'] = [
                allocation_reason(raw_input_df.iloc[idx], financial_need_scores[idx], allocations[idx])
                for idx in range(len(output_df))
            ]

            logger.debug("Computing fairness metrics")
            avg_allocation_male = output_df[output_df['Gender'] == 'Male']['Allocated Amount'].mean()
            avg_allocation_female = output_df[output_df['Gender'] == 'Female']['Allocated Amount'].mean()
            gender_disparity = abs(avg_allocation_male - avg_allocation_female) if not (pd.isna(avg_allocation_male) or pd.isna(avg_allocation_female)) else 0

            avg_allocation_secondary = output_df[output_df['Academic Level'] == 'Secondary School']['Allocated Amount'].mean()
            avg_allocation_university = output_df[output_df['Academic Level'] == 'University']['Allocated Amount'].mean()
            academic_disparity = abs(avg_allocation_secondary - avg_allocation_university) if not (pd.isna(avg_allocation_secondary) or pd.isna(avg_allocation_university)) else 0

            ward_avg_allocations = output_df.groupby('Ward')['Allocated Amount'].mean()
            ward_disparity = ward_avg_allocations.std() if len(ward_avg_allocations) > 1 else 0

            fairness_metrics = {
                'gender_disparity': float(gender_disparity),
                'academic_disparity': float(academic_disparity),
                'ward_disparity': float(ward_disparity)
            }

            logger.debug("Generating output file")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"allocation_output_{timestamp}.xlsx"
            output_df.to_excel(output_filename, index=False)

            logger.debug("Computing final statistics")
            total_allocated = sum(allocations)
            budget_utilization = (total_allocated / budget) * 100 if budget > 0 else 0
            # Fallback to 0 if remaining_budget is None
            remaining_budget_value = remaining_budget if remaining_budget is not None else (budget - total_allocated)
            if abs(remaining_budget_value) < 0.01:
                remaining_budget_value = 0
            stats = {
                'total_allocated': float(total_allocated),
                'budget_utilization': float(min(budget_utilization, 100)),
                'remaining_budget': float(remaining_budget_value),
                'total_applicants': len(allocations),
                'average_allocation': float(total_allocated / len(allocations)) if len(allocations) > 0 else 0,
                'highest_allocation': float(max(allocations)) if allocations else 0,
                'lowest_allocation': float(min(allocations)) if allocations else 0,
            }

            if allocations:
                max_allocation_idx = allocations.index(max(allocations))
                highest_allocation_name = output_df.iloc[max_allocation_idx]['Name']
                min_allocation_idx = allocations.index(min(allocations))
                lowest_allocation_name = output_df.iloc[min_allocation_idx]['Name']
            else:
                highest_allocation_name = "N/A"
                lowest_allocation_name = "N/A"

            stats['highest_allocation_name'] = highest_allocation_name
            stats['lowest_allocation_name'] = lowest_allocation_name

            batch.output_filename = output_filename
            batch.total_applicants = len(allocations)
            batch.total_allocated = float(total_allocated)
            batch.budget_utilization = float(min(budget_utilization, 100))
            batch.fairness_metrics_json = json.dumps(fairness_metrics)
            db.session.commit()

            logger.debug("Returning response")
            return jsonify({
                'filename': output_filename,
                'stats': stats,
                'fairness_metrics': fairness_metrics,
                'validation_warnings': validation_warnings,
                'status': 'success'
            }), 200
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in batch_allocation endpoint: {str(e)}", exc_info=True)
            return jsonify({'error': str(e), 'status': 'error'}), 500

    return render_template('batch_allocation.html')

# Download endpoint
@app.route('/download/<filename>')
@login_required
def download_file(filename):
    try:
        if current_user.role != 'admin':
            return jsonify({'error': 'Unauthorized access.', 'status': 'error'}), 403
        return send_file(resolve_output_file(filename), as_attachment=True)
    except Exception as e:
        logger.error(f"Error in download_file endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/allocation_history')
@login_required
def allocation_history():
    if current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    batches = Batch.query.order_by(Batch.created_at.desc()).all()
    history = []
    for batch in batches:
        allocations = Allocation.query.filter_by(batch_id=batch.id).all()
        total_allocated = batch.total_allocated if batch.total_allocated else sum(a.allocated_amount or 0 for a in allocations)
        total_applicants = batch.total_applicants if batch.total_applicants else len(allocations)
        utilization = batch.budget_utilization if batch.budget_utilization else ((total_allocated / batch.budget) * 100 if batch.budget else 0)
        try:
            fairness_metrics = json.loads(batch.fairness_metrics_json) if batch.fairness_metrics_json else {}
        except json.JSONDecodeError:
            fairness_metrics = {}
        history.append({
            'batch_id': batch.batch_id,
            'created_at': batch.created_at,
            'source_filename': batch.source_filename or 'N/A',
            'output_filename': batch.output_filename,
            'budget': batch.budget,
            'total_applicants': total_applicants,
            'total_allocated': total_allocated,
            'budget_utilization': utilization,
            'gender_disparity': fairness_metrics.get('gender_disparity', 0),
            'academic_disparity': fairness_metrics.get('academic_disparity', 0),
            'ward_disparity': fairness_metrics.get('ward_disparity', 0)
        })

    return render_template('allocation_history.html', history=history)

# Applicant Applications Page
@app.route('/applications')
@login_required
def applications():
    if current_user.role != 'applicant':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    all_applications = Application.query.filter_by(user_id=current_user.id).order_by(Application.submitted_at.desc()).all()
    return render_template('applications.html', all_applications=all_applications)

# Application Form Submission
@app.route('/submit_application', methods=['GET', 'POST'])
@login_required
def submit_application():
    if current_user.role not in ['applicant', 'admin']:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    global remaining_budget, original_budget

    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            logger.debug(f"Received form data: {data}")

            try:
                year_of_application = parse_int(data.get('year_of_application', 2025), 2025)
                course_duration = parse_int(data.get('course_duration', 0))
                expected_year_of_completion = parse_int(data.get('expected_year_of_completion', 2025), 2025)
                amount_applied = parse_number(data.get('amount_applied', 0))
                fee_balance = parse_number(data.get('fee_balance', 0))
                other_support_amount = parse_number(data.get('other_support_amount', 0))
                last_support_year = data.get('last_support_year', '0')
                last_support_year = parse_int(last_support_year)
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting form data to numeric types: {str(e)}")
                flash(f'Error submitting application: Invalid numeric input. Please ensure all numeric fields are valid.', 'danger')
                return redirect(url_for('submit_application'))

            if current_user.role == 'applicant':
                existing_application = Application.query.filter_by(user_id=current_user.id, status='pending').first()
                if existing_application:
                    flash('You already have a pending application. Please wait for it to be processed.', 'warning')
                    return redirect(url_for('applications'))
                user_id = current_user.id
                name = data.get('Name', current_user.name)
            else:
                name = data.get('Name', '')
                email = f"manual_{int(time.time())}@example.com"
                existing_user = User.query.filter_by(email=email).first()
                if not existing_user:
                    user = User(
                        email=email,
                        name=name,
                        gender=data.get('Gender', ''),
                        ward=data.get('Ward', ''),
                        role='applicant'
                    )
                    user.set_password('placeholder')
                    db.session.add(user)
                    db.session.commit()
                else:
                    user = existing_user
                user_id = user.id

            application = Application(
                user_id=user_id,
                name=name,
                academic_level=data.get('academic_level', ''),
                year_of_application=year_of_application,
                average_academic_performance=data.get('average_academic_performance', ''),
                school=data.get('school', ''),
                course_of_study=data.get('course_of_study', ''),
                course_duration=course_duration,
                mode_of_study=data.get('mode_of_study', ''),
                expected_year_of_completion=expected_year_of_completion,
                amount_applied=amount_applied,
                fee_balance=fee_balance,
                family_status=data.get('family_status', ''),
                care_of=data.get('care_of', ''),
                employment_type=data.get('employment_type', ''),
                is_main_source_of_income=data.get('is_main_source_of_income', ''),
                past_ngcdf_support=data.get('past_ngcdf_support', ''),
                other_support=data.get('other_support', ''),
                other_support_amount=other_support_amount,
                last_support_year=last_support_year,
                individual_disability=data.get('individual_disability', ''),
                parent_disability=data.get('parent_disability', ''),
                documents_available=data.get('documents_available', ''),
                recommendation=data.get('recommendation', ''),
                status='pending'
            )
            db.session.add(application)
            db.session.commit()
            logger.debug(f"Application created with ID: {application.id}")

            mapped_data = {}
            mapped_data['acad_level_University'] = 1 if application.academic_level == 'University' else 0
            acad_perf = application.average_academic_performance or 'Poor'
            mapped_data['acad_perf_Good'] = 1 if acad_perf == 'Good' else 0
            mapped_data['acad_perf_Fair'] = 1 if acad_perf == 'Fair' else 0
            mapped_data['acad_perf_Poor'] = 1 if acad_perf == 'Poor' else 0
            mapped_data['course_dur'] = application.course_duration
            mode_of_study = application.mode_of_study or 'Boarding'
            mapped_data['study_mode_Boarding'] = 1 if mode_of_study in ['Boarding', 'Bording'] else 0
            mapped_data['study_mode_Day scholar'] = 1 if mode_of_study == 'Day scholar' else 0
            mapped_data['study_mode_Government Sponsored'] = 1 if mode_of_study == 'Government Sponsored' else 0
            mapped_data['study_mode_Self sponsored'] = 1 if mode_of_study == 'Self sponsored' else 0
            mapped_data['exp_completion'] = application.expected_year_of_completion
            mapped_data['amt_applied'] = application.amount_applied
            mapped_data['fee_balance'] = application.fee_balance
            family_status = application.family_status or 'Other'
            mapped_data['family_status_Total Orphan'] = 1 if family_status == 'Total Orphan' else 0
            mapped_data['family_status_Partial orphan'] = 1 if family_status == 'Partial orphan' else 0
            mapped_data['family_status_Single Parent'] = 1 if family_status == 'Single Parent' else 0
            caregiver = application.care_of or 'Mother'
            mapped_data['caregiver_Guardian'] = 1 if caregiver == 'Guardian' else 0
            mapped_data['caregiver_Father'] = 1 if caregiver == 'Father' else 0
            mapped_data['caregiver_Mother'] = 1 if caregiver == 'Mother' else 0
            emp_type = application.employment_type or 'Unknown'
            mapped_data['emp_type_Contractual'] = 1 if emp_type == 'Contractual' else 0
            mapped_data['emp_type_Parmanent'] = 1 if emp_type == 'Permanent' else 0
            mapped_data['emp_type_Retired'] = 1 if emp_type == 'Retired' else 0
            mapped_data['emp_type_Self Employed'] = 1 if emp_type == 'Self Employed' else 0
            mapped_data['emp_type_Unknown'] = 1 if emp_type == 'Unknown' else 0
            mapped_data['is_main_source'] = application.is_main_source_of_income or 'Yes'
            past_ngcdf = application.past_ngcdf_support or 'No'
            mapped_data['past_ngcdf_Yes'] = 1 if past_ngcdf == 'Yes' else 0
            other_support = application.other_support or 'No'
            mapped_data['other_support_Yes'] = 1 if other_support == 'Yes' else 0
            mapped_data['support_amt'] = application.other_support_amount
            current_year = 2025
            past_support_impact = max(0, 1 - (current_year - application.last_support_year) / 10) if application.last_support_year > 0 else 0
            mapped_data['past_support_impact'] = past_support_impact
            ind_disability = application.individual_disability or 'No'
            mapped_data['ind_disability_Yes'] = 1 if ind_disability == 'Yes' else 0
            pg_disability = application.parent_disability or 'No'
            mapped_data['pg_disability_No'] = 1 if pg_disability == 'No' else 0
            mapped_data['pg_disability_Yes'] = 1 if pg_disability == 'Yes' else 0
            mapped_data['pg_disability_Chronic Disease'] = 1 if pg_disability == 'Chronic Disease' else 0
            mapped_data['pg_disability_Both'] = 1 if pg_disability == 'Both' else 0
            docs_available = application.documents_available or 'No'
            mapped_data['docs_available_No'] = 1 if docs_available == 'No' else 0
            mapped_data['docs_available_Partial'] = 1 if docs_available == 'Partial' else 0
            mapped_data['docs_available_Yes'] = 1 if docs_available == 'Yes' else 0
            recommendation = application.recommendation or 'Approve'
            mapped_data['recommendation_Decline'] = 1 if recommendation == 'Decline' else 0
            vuln_index = (
                mapped_data['ind_disability_Yes'] +
                mapped_data['pg_disability_Yes'] +
                mapped_data['pg_disability_Chronic Disease'] +
                mapped_data['pg_disability_Both'] +
                mapped_data['family_status_Total Orphan'] +
                mapped_data['family_status_Partial orphan'] +
                mapped_data['family_status_Single Parent']
            ) / 7.0
            mapped_data['vuln_index'] = vuln_index
            years_remaining = max(0, expected_year_of_completion - year_of_application)
            mapped_data['years_remaining'] = years_remaining
            course_completion_ratio = (course_duration - years_remaining) / course_duration if course_duration > 0 else 0
            mapped_data['course_completion_ratio'] = max(0, min(1, course_completion_ratio))
            fee_balance = mapped_data['fee_balance']
            log_financial_burden = np.log1p(fee_balance) / np.log1p(1000000) if fee_balance > 0 else 0
            mapped_data['log_financial_burden'] = log_financial_burden
            remaining_fee = max(0, fee_balance - mapped_data['support_amt'])
            mapped_data['remaining_fee'] = remaining_fee

            input_df = pd.DataFrame([mapped_data])
            pca_input_df = input_df[pca_features]
            X_pca = pca.transform(pca_input_df)
            X_pca = np.nan_to_num(X_pca, nan=0.0)
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
            input_df_pca = pd.concat([input_df[['amt_applied', 'course_completion_ratio', 'acad_perf_Good', 'past_support_impact', 'log_financial_burden', 'remaining_fee']], pca_df], axis=1)
            score_context_features = [
                'ind_disability_Yes', 'pg_disability_Yes', 'pg_disability_Chronic Disease',
                'pg_disability_Both', 'family_status_Total Orphan', 'family_status_Partial orphan',
                'family_status_Single Parent', 'past_ngcdf_Yes', 'course_dur',
                'docs_available_No', 'emp_type_Contractual', 'emp_type_Unknown',
                'study_mode_Boarding', 'acad_level_University'
            ]
            for feature in score_context_features:
                input_df_pca[feature] = input_df.get(feature, 0)
            logger.debug(f"Columns in input_df_pca before adding predicted_financial_need_score: {list(input_df_pca.columns)}")

            # Compute financial need score components
            input_df_pca['vulnerability_score'] = (
                3 * input_df_pca.get('ind_disability_Yes', 0) +
                2 * input_df_pca.get('pg_disability_Yes', 0) +
                2 * input_df_pca.get('pg_disability_Chronic Disease', 0) +
                4 * input_df_pca.get('pg_disability_Both', 0) +
                3 * input_df_pca.get('family_status_Total Orphan', 0) +
                2 * input_df_pca.get('family_status_Partial orphan', 0) +
                1 * input_df_pca.get('family_status_Single Parent', 0) +
                1 * (1 - input_df_pca['past_ngcdf_Yes']) +
                np.log1p(input_df_pca['remaining_fee']) / np.log1p(input_df_pca['remaining_fee'].max()) * 5 +
                (input_df_pca['course_completion_ratio'] / input_df_pca['course_completion_ratio'].max()) * 2 +
                (1 - input_df_pca['course_dur'] / 4) * 2 +
                1 * input_df_pca.get('docs_available_No', 0) +
                2 * input_df_pca.get('emp_type_Contractual', 0) +
                1 * input_df_pca.get('emp_type_Unknown', 0) +
                1 * input_df_pca.get('study_mode_Boarding', 0) +
                2 * input_df_pca.get('acad_level_University', 0)
            ).fillna(0)
            max_vuln = input_df_pca['vulnerability_score'].max()
            if max_vuln > 0:
                input_df_pca['vulnerability_score'] = (input_df_pca['vulnerability_score'] / max_vuln) * 10

            # Apply Yeo-Johnson transformation
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            features_for_score = ['remaining_fee', 'vulnerability_score', 'log_financial_burden', 
                                 'course_completion_ratio', 'past_support_impact']
            for feature in ['remaining_fee', 'vulnerability_score', 'log_financial_burden']:
                input_df_pca[f'{feature}_transformed'] = pt.fit_transform(input_df_pca[[feature]])
            features_for_score = [f'{feature}_transformed' if feature in ['remaining_fee', 'vulnerability_score', 'log_financial_burden'] else feature for feature in features_for_score]

            # Standardize features
            scaler_std = StandardScaler()
            input_df_pca[features_for_score] = scaler_std.fit_transform(input_df_pca[features_for_score])

            # Learn weights for financial need score
            X_score = input_df_pca[features_for_score].fillna(0)
            y_score = input_df_pca['amt_applied'].fillna(input_df_pca['amt_applied'].median())
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_score, y_score)
            learned_weights = pd.Series(ridge_model.coef_, index=features_for_score).to_dict()
            logger.debug(f"Learned Financial Need Score Weights: {learned_weights}")

            # Calculate financial need score
            input_df_pca['financial_need_score'] = (
                sum(learned_weights[feature] * input_df_pca[feature] for feature in features_for_score) +
                0.1 * (input_df_pca['remaining_fee_transformed'] * input_df_pca['vulnerability_score_transformed'])
            )

            # Standardize before sigmoid transformation
            input_df_pca['financial_need_score'] = (input_df_pca['financial_need_score'] - input_df_pca['financial_need_score'].mean()) / input_df_pca['financial_need_score'].std()

            # Apply sigmoid transformation
            input_df_pca['financial_need_score'] = expit(input_df_pca['financial_need_score'])

            # Scale to 0-100 with clipping
            min_score = input_df_pca['financial_need_score'].min()
            max_score = input_df_pca['financial_need_score'].max()
            if max_score > min_score:
                input_df_pca['financial_need_score'] = 100 * (input_df_pca['financial_need_score'] - min_score) / (max_score - min_score)
            else:
                input_df_pca['financial_need_score'] = 0
            input_df_pca['financial_need_score'] = input_df_pca['financial_need_score'].clip(lower=1, upper=99)

            financial_need_score = input_df_pca['financial_need_score'].iloc[0]

            # Predict financial need score using the meta-model
            scaler_meta = MinMaxScaler()
            model_input = scaler_meta.fit_transform(input_df_pca[model_input_features])
            model_input_df = pd.DataFrame(model_input, columns=model_input_features)

            # Debug: Log the number of features expected by the model and provided
            logger.debug(f"Number of features expected by RandomForestRegressor: {rf_model.n_features_in_}")
            logger.debug(f"Number of features in model_input: {model_input.shape[1]}")
            logger.debug(f"Features provided: {model_input_features}")

            rf_pred = np.nan_to_num(rf_model.predict(model_input), nan=0.0)
            xgb_pred = np.nan_to_num(xgb_model.predict(model_input), nan=0.0)
            lgb_pred = np.nan_to_num(lgb_model.predict(model_input), nan=0.0)
            meta_features = np.column_stack((rf_pred, xgb_pred, lgb_pred))
            predicted_financial_need_score = meta_model.predict(meta_features)[0]
            predicted_financial_need_score = np.nan_to_num(predicted_financial_need_score, nan=0.0)
            predicted_financial_need_score = max(0, min(1, predicted_financial_need_score))
            if predicted_financial_need_score >= 0.98 or predicted_financial_need_score <= 0.01:
                predicted_financial_need_score = max(0.01, min(0.99, financial_need_score / 100.0))

            input_df_pca['predicted_financial_need_score'] = predicted_financial_need_score
            logger.debug(f"Columns in input_df_pca after adding predicted_financial_need_score: {list(input_df_pca.columns)}")

            missing_features = [feature for feature in state_features if feature not in input_df_pca.columns]
            if missing_features:
                logger.warning(f"Missing features in input_df_pca: {missing_features}")
                for feature in missing_features:
                    input_df_pca[feature] = 0

            state_df = input_df_pca[state_features]
            logger.debug(f"Columns in state_df before scaling: {list(state_df.columns)}")
            states = build_state_data(state_df)
            logger.debug(f"State shape for single allocation: {states.shape}")

            if states.shape[1] != len(state_features):
                logger.error(f"State has {states.shape[1]} features, but expects {len(state_features)} features")
                raise ValueError(f"State has {states.shape[1]} features, but expects {len(state_features)} features")

            # Calculate budget_to_applied_ratio for single allocation
            total_applied_amount = input_df_pca['amt_applied'].sum()
            budget = 50000  # Default budget for single allocation if not set
            if original_budget is not None:
                budget = original_budget
            budget_to_applied_ratio = budget / total_applied_amount if total_applied_amount > 0 else 1.0
            logger.debug(f"Total Applied Amount: {total_applied_amount} KSh, Budget: {budget} KSh, Budget-to-Applied Ratio: {budget_to_applied_ratio:.4f}")

            if remaining_budget is None:
                remaining_budget = budget

            if remaining_budget <= 0:
                allocation = 0
            else:
                # Initialize DDPG environment and agent for single allocation
                env = BursaryAllocationEnv(states, np.array([application.amount_applied]), np.array([predicted_financial_need_score]), budget)
                agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim, max_action=1.0)

                # Load the trained DDPG model
                try:
                    agent.actor.load_state_dict(torch.load('ddpg_actor.pth', map_location=torch.device('cpu')))
                    logger.debug("Successfully loaded DDPG actor model from 'ddpg_actor.pth' for single allocation")
                except Exception as e:
                    logger.error(f"Failed to load DDPG actor model: {str(e)}")
                    raise Exception(f"Failed to load DDPG actor model: {str(e)}")

                # Allocate using DDPG agent
                env.reset()
                agent.actor.eval()
                with torch.no_grad():
                    state = env.state_space_np[0]
                    applied_amount = env.amt_applied_np[0]
                    already_allocated = env.allocations[0]
                    remaining_applied = max(0, applied_amount - already_allocated)
                    need_score = env.need_scores_np[0]
                    policy_action = float(np.ravel(agent.select_action(state))[0])
                    adjusted_percentage = env.budget_to_applied_ratio * need_score * policy_action
                    adjusted_percentage = min(adjusted_percentage, 0.4)  # Cap at 40% to prevent excessive single allocations
                    adjusted_percentage = np.clip(adjusted_percentage, 0, 1)

                    # Ensure minimum allocation percentages
                    if need_score > 0.5:
                        adjusted_percentage = max(adjusted_percentage, 0.3)  # Minimum 30% for high-need students
                    else:
                        adjusted_percentage = max(adjusted_percentage, 0.2)  # Minimum 20% for non-high-need students

                    allocation = remaining_applied * adjusted_percentage
                    if allocation > 0:
                        allocation = max(allocation, 500)  # Minimum allocation for single application
                    allocation = min(allocation, remaining_applied)
                    if allocation > remaining_budget:
                        allocation = 0
                    remaining_budget -= allocation

            # Create a temporary batch for single allocation
            batch_id_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch = Batch(batch_id=batch_id_str, budget=budget)
            db.session.add(batch)
            db.session.commit()
            batch_id = batch.id

            allocation_record = Allocation(
                application_id=application.id,
                batch_id=batch_id,
                financial_need_score=predicted_financial_need_score,
                allocated_amount=allocation
            )
            db.session.add(allocation_record)
            db.session.commit()
            logger.debug(f"Allocation created: ID={allocation_record.id}, Application ID={allocation_record.application_id}, Amount={allocation_record.allocated_amount}")

            saved_allocation = Allocation.query.filter_by(application_id=application.id).first()
            if saved_allocation:
                logger.debug(f"Verified allocation in database: ID={saved_allocation.id}, Amount={saved_allocation.allocated_amount}")
            else:
                logger.error("Allocation not found in database after commit")
                flash('Allocation was created but could not be verified in the database. Please contact support.', 'warning')

            flash(f'Application submitted successfully! Application ID: {application.id}. Awaiting admin review.', 'success')

            # Redirect based on user role
            if current_user.role == 'applicant':
                return redirect(url_for('applications'))
            else:
                try:
                    logger.debug(f"Registered endpoints: {[rule.endpoint for rule in app.url_map.iter_rules()]}")
                    return redirect(url_for('single_allocation'))
                except Exception as e:
                    logger.error(f"Error redirecting to single_allocation: {str(e)}")
                    flash('Error redirecting to single allocation page. Redirecting to dashboard.', 'warning')
                    return redirect(url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in submit_application endpoint: {str(e)}", exc_info=True)
            flash(f'Error submitting application: {str(e)}', 'danger')
            return redirect(url_for('submit_application'))

    try:
        return render_template('application_form.html')
    except Exception as e:
        logger.error(f"Error rendering application_form.html: {str(e)}", exc_info=True)
        return jsonify({'error': 'application_form.html', 'status': 'error', 'details': str(e)}), 500

# Single Allocation Page for Admins
@app.route('/single_allocation', methods=['GET', 'POST'])
@login_required
def single_allocation():
    if current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    all_applications = Application.query.order_by(Application.submitted_at.desc()).limit(500).all()
    applications_data = []
    for app in all_applications:
        user = User.query.get(app.user_id)
        allocation = Allocation.query.filter_by(application_id=app.id).first()
        applications_data.append({
            'id': app.id,
            'name': app.name or (user.name if user else 'Unknown'),
            'academic_level': app.academic_level,
            'amount_applied': app.amount_applied,
            'financial_need_score': allocation.financial_need_score * 100 if allocation else 0,
            'suggested_allocation': allocation.allocated_amount if allocation else 0,
            'status': app.status
        })

    if request.method == 'POST' and 'action' in request.form:
        try:
            action = request.form.get('action')
            application_id = int(request.form.get('application_id'))
            application = Application.query.get_or_404(application_id)

            if action == 'approve':
                application.status = 'approved'
                flash('Application approved successfully.', 'success')
            elif action == 'decline':
                application.status = 'declined'
                flash('Application declined.', 'info')

            db.session.commit()
            return redirect(url_for('single_allocation'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in single_allocation endpoint: {str(e)}", exc_info=True)
            flash(f'Error processing application: {str(e)}', 'danger')
            return redirect(url_for('single_allocation'))

    return render_template('single_allocation.html', applications=applications_data)

# Endpoint for Applicants to Check Application Status
@app.route('/application_status', methods=['GET'])
@login_required
def application_status():
    if current_user.role != 'applicant':
        return jsonify({'error': 'Unauthorized access.', 'status': 'error'}), 403

    application_id = request.args.get('application_id', type=int)
    if application_id:
        application = Application.query.filter_by(id=application_id, user_id=current_user.id).first()
        if not application:
            return jsonify({'status': 'No application found', 'application_id': application_id, 'allocated_amount': 0})
    else:
        application = Application.query.filter_by(user_id=current_user.id).order_by(Application.submitted_at.desc()).first()
        if not application:
            return jsonify({'status': 'No application found', 'application_id': 0, 'allocated_amount': 0})

    allocation = Allocation.query.filter_by(application_id=application.id).first()
    return jsonify({
        'status': application.status,
        'application_id': application.id,
        'allocated_amount': allocation.allocated_amount if allocation else 0
    })

# Favicon handler
@app.route('/favicon.ico')
def favicon():
    return '', 404

# Profile endpoint
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.email = request.form.get('email')
        current_user.name = request.form.get('name')
        current_user.gender = request.form.get('gender')
        current_user.ward = request.form.get('ward')
        
        password = request.form.get('password')
        if password:
            current_user.set_password(password)
        
        try:
            db.session.commit()
            flash('Profile updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'danger')
        
        return redirect(url_for('profile'))

    return render_template('profile.html')

def get_or_create_user_setting(user):
    setting = UserSetting.query.filter_by(user_id=user.id).first()
    if setting:
        return setting
    setting = UserSetting(user_id=user.id)
    db.session.add(setting)
    db.session.commit()
    return setting

@app.route('/thank_yous', methods=['GET', 'POST'])
@login_required
def thank_yous():
    if request.method == 'POST':
        action = request.form.get('action', 'create')
        if action == 'mark_reviewed' and current_user.role == 'admin':
            message = ThankYouMessage.query.get(request.form.get('message_id'))
            if message:
                message.reviewed = True
                db.session.commit()
                flash('Thank-you message marked as reviewed.', 'success')
            return redirect(url_for('thank_yous'))

        title = request.form.get('title', '').strip()
        body = request.form.get('message', '').strip()
        visibility = request.form.get('visibility', 'private')
        application_id = request.form.get('application_id') or None

        if not title or not body:
            flash('Please add both a title and a message.', 'danger')
            return redirect(url_for('thank_yous'))

        if visibility not in ['private', 'public']:
            visibility = 'private'

        if application_id:
            application = Application.query.get(application_id)
            if not application or (current_user.role == 'applicant' and application.user_id != current_user.id):
                flash('Please choose one of your own applications.', 'danger')
                return redirect(url_for('thank_yous'))

        db.session.add(ThankYouMessage(
            user_id=current_user.id,
            application_id=application_id,
            title=title,
            message=body,
            visibility=visibility
        ))
        db.session.commit()
        flash('Thank-you message submitted successfully.', 'success')
        return redirect(url_for('thank_yous'))

    if current_user.role in ['admin', 'stakeholder']:
        messages = ThankYouMessage.query.order_by(ThankYouMessage.created_at.desc()).limit(100).all()
        applications = Application.query.order_by(Application.submitted_at.desc()).limit(100).all()
    else:
        messages = ThankYouMessage.query.filter(
            or_(ThankYouMessage.user_id == current_user.id, ThankYouMessage.visibility == 'public')
        ).order_by(ThankYouMessage.created_at.desc()).limit(100).all()
        applications = Application.query.filter_by(user_id=current_user.id).order_by(Application.submitted_at.desc()).all()

    stats = {
        'total': len(messages),
        'public': sum(1 for item in messages if item.visibility == 'public'),
        'pending_review': sum(1 for item in messages if not item.reviewed),
    }
    return render_template('thank_yous.html', messages=messages, applications=applications, stats=stats)

@app.route('/communications', methods=['GET', 'POST'])
@login_required
def communications():
    if request.method == 'POST':
        subject = request.form.get('subject', '').strip()
        body = request.form.get('body', '').strip()
        audience = request.form.get('audience', 'staff' if current_user.role == 'applicant' else 'all')
        priority = request.form.get('priority', 'normal')

        if not subject or not body:
            flash('Please add both a subject and a message.', 'danger')
            return redirect(url_for('communications'))

        if current_user.role == 'applicant':
            audience = 'staff'
            status = 'submitted'
        else:
            status = 'sent'
            if audience not in ['all', 'applicant', 'stakeholder', 'admin']:
                audience = 'all'

        if priority not in ['normal', 'high', 'urgent']:
            priority = 'normal'

        db.session.add(CommunicationMessage(
            sender_id=current_user.id,
            subject=subject,
            body=body,
            audience=audience,
            priority=priority,
            status=status
        ))
        db.session.commit()
        flash('Communication saved successfully.', 'success')
        return redirect(url_for('communications'))

    if current_user.role in ['admin', 'stakeholder']:
        messages = CommunicationMessage.query.order_by(CommunicationMessage.created_at.desc()).limit(120).all()
    else:
        messages = CommunicationMessage.query.filter(
            or_(
                CommunicationMessage.sender_id == current_user.id,
                CommunicationMessage.audience == 'all',
                CommunicationMessage.audience == current_user.role
            )
        ).order_by(CommunicationMessage.created_at.desc()).limit(120).all()

    stats = {
        'total': len(messages),
        'urgent': sum(1 for item in messages if item.priority == 'urgent'),
        'submitted': sum(1 for item in messages if item.status == 'submitted'),
    }
    return render_template('communications.html', messages=messages, stats=stats)

@app.route('/support', methods=['GET', 'POST'])
@login_required
def support():
    if request.method == 'POST':
        action = request.form.get('action', 'create')
        if action == 'update_status' and current_user.role == 'admin':
            ticket = SupportTicket.query.get(request.form.get('ticket_id'))
            new_status = request.form.get('status', 'open')
            if ticket and new_status in ['open', 'in_progress', 'resolved']:
                ticket.status = new_status
                db.session.commit()
                flash('Support ticket updated.', 'success')
            return redirect(url_for('support'))

        title = request.form.get('title', '').strip()
        message = request.form.get('message', '').strip()
        category = request.form.get('category', 'General')
        priority = request.form.get('priority', 'normal')

        if not title or not message:
            flash('Please add both a ticket title and details.', 'danger')
            return redirect(url_for('support'))

        if priority not in ['normal', 'high', 'urgent']:
            priority = 'normal'

        db.session.add(SupportTicket(
            user_id=current_user.id,
            title=title,
            message=message,
            category=category,
            priority=priority
        ))
        db.session.commit()
        flash('Support ticket created successfully.', 'success')
        return redirect(url_for('support'))

    if current_user.role == 'admin':
        tickets = SupportTicket.query.order_by(SupportTicket.created_at.desc()).limit(120).all()
    else:
        tickets = SupportTicket.query.filter_by(user_id=current_user.id).order_by(SupportTicket.created_at.desc()).limit(120).all()

    stats = {
        'open': sum(1 for item in tickets if item.status == 'open'),
        'in_progress': sum(1 for item in tickets if item.status == 'in_progress'),
        'resolved': sum(1 for item in tickets if item.status == 'resolved'),
    }
    return render_template('support.html', tickets=tickets, stats=stats)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    setting = get_or_create_user_setting(current_user)
    if request.method == 'POST':
        current_user.name = request.form.get('name', current_user.name).strip()
        current_user.gender = request.form.get('gender', current_user.gender)
        current_user.ward = request.form.get('ward', current_user.ward)
        setting.email_notifications = request.form.get('email_notifications') == 'on'
        setting.sms_notifications = request.form.get('sms_notifications') == 'on'
        setting.compact_tables = request.form.get('compact_tables') == 'on'
        setting.theme = request.form.get('theme', 'modern')
        setting.default_budget = parse_number(request.form.get('default_budget'), setting.default_budget or 1000000)

        password = request.form.get('password', '').strip()
        if password:
            current_user.set_password(password)

        try:
            db.session.commit()
            flash('Settings saved successfully.', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error saving settings: {str(e)}', 'danger')
        return redirect(url_for('settings'))

    return render_template('settings.html', setting=setting)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080)
