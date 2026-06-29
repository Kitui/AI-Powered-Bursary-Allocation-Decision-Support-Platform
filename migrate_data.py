from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import joblib
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bursary.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Models (same as in app.py)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='applicant')
    name = db.Column(db.String(100))
    gender = db.Column(db.String(20))
    ward = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
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
    status = db.Column(db.String(20), default='submitted')
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

class Allocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    application_id = db.Column(db.Integer, db.ForeignKey('application.id'), nullable=False)
    financial_need_score = db.Column(db.Float)
    allocated_amount = db.Column(db.Float)
    allocated_at = db.Column(db.DateTime, default=datetime.utcnow)

def migrate_data():
    try:
        # Load the existing data
        data_file = '/app/data/latest_allocation_data.joblib'
        if not os.path.exists(data_file):
            print("No existing data to migrate.")
            return

        latest_allocation_data = joblib.load(data_file)
        if not latest_allocation_data or 'input_df' not in latest_allocation_data:
            print("Invalid or empty data file.")
            return

        input_df = latest_allocation_data['input_df']
        print(f"Found {len(input_df)} records to migrate.")

        # Migrate data
        with app.app_context():
            for idx, row in input_df.iterrows():
                # Create a user
                user = User(
                    email=f"migrated_{idx}@example.com",
                    password="placeholder",
                    name=row.get('Name', ''),
                    gender=row.get('Gender', ''),
                    ward=row.get('Ward', '')
                )
                db.session.add(user)
                db.session.flush()  # Ensure user ID is generated

                # Create an application
                application = Application(
                    user_id=user.id,
                    academic_level=row.get('Academic Level', ''),
                    year_of_application=int(row.get('Year of Application', 2025)),
                    average_academic_performance=row.get('Average Academic Performance', ''),
                    school=row.get('School', ''),
                    course_of_study=row.get('Course of Study', ''),
                    course_duration=int(row.get('Course Duration', 0)),
                    mode_of_study=row.get('Mode of Study', ''),
                    expected_year_of_completion=int(row.get('Expected Year of Course Completion', 2025)),
                    amount_applied=float(row.get('Amount Applied (Kshs)', 0)),
                    fee_balance=float(row.get('Fee Balance', 0)),
                    family_status=row.get('Family status', ''),
                    care_of=row.get('Care of', ''),
                    employment_type=row.get('Employment Type', ''),
                    is_main_source_of_income=row.get('Is it a main Source of Income()', ''),
                    past_ngcdf_support=row.get('Past Financial Support(NG-CDF)', ''),
                    other_support=row.get('Past Financial Support(Others)', ''),
                    other_support_amount=float(row.get('If yes, specify how much.', 0)),
                    last_support_year=int(row.get('Last Received', 0)),
                    individual_disability=row.get('Individual Disability Status', ''),
                    parent_disability=row.get('Parent/Guardian Disability Status', ''),
                    documents_available=row.get('Supportive documents Available', ''),
                    recommendation=row.get('Recommendation', ''),
                    status='approved' if row.get('allocated_amount', 0) > 0 else 'rejected'
                )
                db.session.add(application)
                db.session.flush()

                # Create an allocation
                allocation = Allocation(
                    application_id=application.id,
                    financial_need_score=row.get('predicted_financial_need_score', 0),
                    allocated_amount=row.get('allocated_amount', 0)
                )
                db.session.add(allocation)

            db.session.commit()
            print("Data migration completed successfully.")

    except Exception as e:
        db.session.rollback()
        print(f"Error during migration: {str(e)}")

if __name__ == '__main__':
    migrate_data()
