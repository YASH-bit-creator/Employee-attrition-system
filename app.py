import streamlit as st
import pandas as pd
import pickle

# Load model
with open("employee_attrition_xgb.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide"
)

st.title("Employee Attrition Prediction System")
st.write(
    "Predict the likelihood of an employee leaving the organization."
)

# --------------------
# MAPPINGS
# --------------------

gender_map = {
    "Female": 0,
    "Male": 1
}

overtime_map = {
    "No": 0,
    "Yes": 1
}

business_map = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2
}

department_map = {
    "Human Resources": 0,
    "Research & Development": 1,
    "Sales": 2
}

marital_map = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

education_map = {
    "Human Resources": 0,
    "Life Sciences": 1,
    "Marketing": 2,
    "Medical": 3,
    "Other": 4,
    "Technical Degree": 5
}

jobrole_map = {
    "Healthcare Representative": 0,
    "Human Resources": 1,
    "Laboratory Technician": 2,
    "Manager": 3,
    "Manufacturing Director": 4,
    "Research Director": 5,
    "Research Scientist": 6,
    "Sales Executive": 7,
    "Sales Representative": 8
}

# --------------------
# INPUTS
# --------------------

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 60, 30)

    BusinessTravel = business_map[
        st.selectbox(
            "Business Travel",
            list(business_map.keys())
        )
    ]

    DailyRate = st.number_input(
        "Daily Rate",
        100,
        2000,
        800
    )

    Department = department_map[
        st.selectbox(
            "Department",
            list(department_map.keys())
        )
    ]

    DistanceFromHome = st.number_input(
        "Distance From Home",
        1,
        50,
        5
    )

    Education = st.slider(
        "Education Level",
        1,
        5,
        3
    )

    EducationField = education_map[
        st.selectbox(
            "Education Field",
            list(education_map.keys())
        )
    ]

    EnvironmentSatisfaction = st.slider(
        "Environment Satisfaction",
        1,
        4,
        3
    )

    Gender = gender_map[
        st.selectbox(
            "Gender",
            list(gender_map.keys())
        )
    ]

    HourlyRate = st.number_input(
        "Hourly Rate",
        20,
        100,
        60
    )

    JobInvolvement = st.slider(
        "Job Involvement",
        1,
        4,
        3
    )

    JobLevel = st.slider(
        "Job Level",
        1,
        5,
        2
    )

with col2:

    JobRole = jobrole_map[
        st.selectbox(
            "Job Role",
            list(jobrole_map.keys())
        )
    ]

    JobSatisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    MaritalStatus = marital_map[
        st.selectbox(
            "Marital Status",
            list(marital_map.keys())
        )
    ]

    MonthlyIncome = st.number_input(
        "Monthly Income",
        1000,
        50000,
        10000
    )

    MonthlyRate = st.number_input(
        "Monthly Rate",
        2000,
        30000,
        15000
    )

    NumCompaniesWorked = st.slider(
        "Companies Worked",
        0,
        10,
        2
    )

    OverTime = overtime_map[
        st.selectbox(
            "OverTime",
            list(overtime_map.keys())
        )
    ]

    PercentSalaryHike = st.slider(
        "Percent Salary Hike",
        10,
        30,
        15
    )

    PerformanceRating = st.slider(
        "Performance Rating",
        1,
        4,
        3
    )

    RelationshipSatisfaction = st.slider(
        "Relationship Satisfaction",
        1,
        4,
        3
    )

    StockOptionLevel = st.slider(
        "Stock Option Level",
        0,
        3,
        1
    )

# Remaining features

TotalWorkingYears = st.slider(
    "Total Working Years",
    0,
    40,
    10
)

TrainingTimesLastYear = st.slider(
    "Training Times Last Year",
    0,
    10,
    3
)

WorkLifeBalance = st.slider(
    "Work Life Balance",
    1,
    4,
    3
)

YearsAtCompany = st.slider(
    "Years At Company",
    0,
    40,
    5
)

YearsInCurrentRole = st.slider(
    "Years In Current Role",
    0,
    20,
    3
)

YearsSinceLastPromotion = st.slider(
    "Years Since Last Promotion",
    0,
    15,
    1
)

YearsWithCurrManager = st.slider(
    "Years With Current Manager",
    0,
    20,
    3
)

# --------------------
# PREDICTION
# --------------------

if st.button("Predict Attrition"):

    input_data = pd.DataFrame([{
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"High Attrition Risk: {probability*100:.2f}%"
        )
    else:
        st.success(
            f"Low Attrition Risk: {probability*100:.2f}%"
        )