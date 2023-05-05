import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *
import streamlit as st

st.title('Evidently Report')

#upload_file

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    glass_data = pd.read_csv(uploaded_file)

# Load your data into a pandas dataframe
# glass_data = pd.read_csv(r"C:\Users\umang\Downloads\KNN_key\KNN_key\Glass.csv")

    # Rename the target variable to "target"
    glass_data.rename(columns={"Type": "target"}, inplace=True)

    # Split the data into reference and current sets
    reference, current = train_test_split(glass_data, test_size=0.5)

    # Train your model on the reference set
    X_ref = reference.drop(columns=["target"])
    y_ref = reference["target"]

    # selectbox for creating options for 
    options=X_ref.columns

    selected_option = st.selectbox('Select an option', options)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_ref, y_ref)

    # Predict the target variable for the current set
    X_cur = current.drop(columns=["target"])
    y_true = current["target"]
    y_pred = knn_model.predict(X_cur)
    y_pred1=knn_model.predict(X_ref)
    # Add the predicted values as a new column to the current set
    current["prediction"] = y_pred
    reference['prediction']=y_pred1
    print(current)
    print(reference)

    report1 = Report(metrics=[
    DataDriftPreset(), 
    ])

    report1.run(reference_data=reference, current_data=current)
    report1

    report = Report(metrics=[
    ColumnSummaryMetric(column_name=selected_option),
    ColumnQuantileMetric(column_name=selected_option, quantile=0.25),
    ColumnDriftMetric(column_name=selected_option),

    ])

    report.run(reference_data=reference, current_data=current)
    report
    report.as_dict()
    report.json()


    tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=reference, current_data=current)
    tests

    suite = TestSuite(tests=[
    NoTargetPerformanceTestPreset(),
    ])

    suite.run(reference_data=reference, current_data=current)
    suite
    suite = TestSuite(tests=[
    TestColumnDrift(selected_option),
    TestShareOfOutRangeValues(selected_option),
    generate_column_tests(TestMeanInNSigmas, columns='num'),

    ])

    suite.run(reference_data=reference, current_data=current)
    suite

    suite.as_dict()
    suite.json()

    if os.path.exists(r'C:\Users\umang\Downloads\KNN_key\KNN_key\test_suite.html'):
        os.remove(r'C:\Users\umang\Downloads\KNN_key\KNN_key\test_suite.html')
    if os.path.exists(r'C:\Users\umang\Downloads\KNN_key\KNN_key\report.html'):
        os.remove(r'C:\Users\umang\Downloads\KNN_key\KNN_key\report.html')
    if os.path.exists(r'C:\Users\umang\Downloads\KNN_key\report1.html'):
        os.remove(r'C:\Users\umang\Downloads\KNN_key\report1.html')

    suite.save_html('test_suite.html')
    report.save_html('report.html')
    report1.save_html('report1.html')


    if st.button("Click me"):

        with open(r'C:\Users\umang\Downloads\KNN_key\report1.html', 'r', encoding='utf-8') as f:
            html = f.read()

        st.components.v1.html(html, width=700, height=500, scrolling=True)


        with open(r'C:\Users\umang\Downloads\KNN_key\report.html', 'r', encoding='utf-8') as f:
            html = f.read()

        st.components.v1.html(html, width=700, height=500, scrolling=True)

        with open(r'C:\Users\umang\Downloads\KNN_key\test_suite.html', 'r', encoding='utf-8') as f:
            html = f.read()

        st.components.v1.html(html, width=700, height=500, scrolling=True)


