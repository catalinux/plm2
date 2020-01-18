import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
          'tab:cyan', 'tab:gray']
MARKERS = ['o', 'v', 's', '<', '>', '8', '^', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


def plot2d(X, y_pred, y_true, mode=None, centroids=None):
    transformer = None
    X_r = X

    if mode is not None:
        transformer = mode(n_components=2)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 2, 'plot2d only works with 2-dimensional data'

    plt.grid()
    for ix, iyp, iyt in zip(X_r, y_pred, y_true):
        plt.plot(ix[0], ix[1],
                 c=COLORS[iyp],
                 marker=MARKERS[iyt])

    if centroids is not None:
        C_r = centroids
        if transformer is not None:
            C_r = transformer.fit_transform(centroids)
        for cx in C_r:
            plt.plot(cx[0], cx[1],
                     marker=MARKERS[-1],
                     markersize=10,
                     c='red')

    plt.show()


def plot3d(X, y_pred, y_true, mode=None, centroids=None):
    transformer = None
    X_r = X
    if mode is not None:
        transformer = mode(n_components=3)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 3, 'plot2d only works with 3-dimensional data'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.elev = 30
    ax.azim = 120

    for ix, iyp, iyt in zip(X_r, y_pred, y_true):
        ax.plot(xs=[ix[0]], ys=[ix[1]], zs=[ix[2]], zdir='z',
                c=COLORS[iyp],
                marker=MARKERS[iyt])

    if centroids is not None:
        C_r = centroids
        if transformer is not None:
            C_r = transformer.fit_transform(centroids)
        for cx in C_r:
            ax.plot(xs=[cx[0]], ys=[cx[1]], zs=[cx[2]], zdir='z',
                    marker=MARKERS[-1],
                    markersize=10,
                    c='red')
    plt.show()


def get_data():
    df = pd.read_csv('dataset_diabetes/diabetic_data.csv', na_values='?')
    print("Shap1 ", df.shape)
    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    print("Shape 2", df.shape)

    df = df.drop(["weight", "payer_code", "medical_specialty"], axis=1)
    df = df.drop(df[df.gender == 'Unknown/Invalid'].index)
    meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    df['n_medication_changes'] = 0
    for i in meds:
        temp_col = str(i) + '_'
        df[temp_col] = df[i].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
        df['n_medication_changes'] = df['n_medication_changes'] + df[temp_col]
        del df[temp_col]
    # 3- n_medications
    # count medications
    for i in meds:
        df[i] = df[i].replace('No', 0)
        df[i] = df[i].replace('Steady', 1)
        df[i] = df[i].replace('Up', 1)
        df[i] = df[i].replace('Down', 1)
    df['n_medications'] = 0
    for i in meds:
        df['n_medications'] = df['n_medications'] + df[i]

    df = df.drop(meds, axis=1)

    df['readmitted'] = df['readmitted'].replace('>30', 0)
    df['readmitted'] = df['readmitted'].replace('<30', 1)
    df['readmitted'] = df['readmitted'].replace('NO', 0)
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
                "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
    df['age'] = df.age.map(age_dict)
    df['age'] = df['age'].astype('int64')

    df['diabetesMed'] = np.where(df['diabetesMed'] == "Yes", 1, 0)

   # df = df.drop(df['discharge_disposition_id'].isin([13, 14, 11, 19, 20, 21]).index)

    # 11, 19, 20, 21     mean    the     patient    died
    df['admission_type_id'].value_counts()
    df['admission_type_id'] = df['admission_type_id'].replace([8], [6])
    df['admission_type_id'] = df['admission_type_id'].replace([6], [5])

    replacelist = ['home', 'hospital', 'nursing', 'nursing', 'hospice', 'hhealth', 'leftAMA', 'hhealth', 'hospital',
                   'hospital',
                   'died', 'hospital', 'hospice', 'hospice', 'hospital', 'outpatient', 'outpatient', 'unknown', 'died',
                   'died',
                   'died', 'outpatient', 'hospital', 'nursing', 'unknown', 'unknown', 'nursing', 'psych', 'hospital',
                   'outpatient']

    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(list(range(1, 31)), replacelist)
    df["admission"] = df["admission_source_id"].map({1: " Physician Referral",
                                                     2: "Clinic Referral",
                                                     3: "HMO Referral",
                                                     4: "Transfer from a hospital",
                                                     5: " Transfer from a Skilled Nursing Facility (SNF)",
                                                     6: " Transfer from another health care facility",
                                                     7: " Emergency Room",
                                                     8: " Court/Law Enforcement",
                                                     9: " Not Available",
                                                     10: " Transfer from critial access hospital",
                                                     11: "Normal Delivery",
                                                     12: " Premature Delivery",
                                                     13: " Sick Baby",
                                                     14: " Extramural Birth",
                                                     15: "Not Available",
                                                     17: "NULL",
                                                     18: " Transfer From Another Home Health Agency",
                                                     19: "Readmission to Same Home Health Agency",
                                                     20: " Not Mapped",
                                                     21: "Unknown/Invalid",
                                                     22: " Transfer from hospital inpt/same fac reslt in a sep claim",
                                                     23: " Born inside this hospital",
                                                     24: " Born outside this hospital",
                                                     25: " Transfer from Ambulatory Surgery Center",
                                                     26: "Transfer from Hospice"});

    df.drop(['admission_type_id', 'encounter_id', 'patient_nbr'], axis=1, inplace=True)
    s = df["admission_source_id"].value_counts()
    df['admission_source_id'].isin(s.index[s >= 30]).index
    # df = df.drop(['discharge_disposition_id', 'admission_source_id'], axis=1)
    df = df[df["service_utilization"] < 11]

    return df


def get_unclean_data():
    df = pd.read_csv('dataset_diabetes/diabetic_data.csv', na_values='?')
    df.head()
    df_missing = df.copy()
    missing = df_missing.isnull().sum()
    print(missing)
    missing_rates = np.round(100 * missing[missing > 0] / df.__len__(), 3).to_clipboard()

    df = df.drop(["weight", "payer_code", "medical_specialty", 'encounter_id', 'admission_type_id',
                  'discharge_disposition_id', 'admission_source_id'], axis=1)

    df = df.drop(df[df.gender == 'Unknown/Invalid'].index)
    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    df = df.drop(meds, axis=1)
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
                "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
    df['age'] = df.age.map(age_dict)
    df['age'] = df['age'].astype('int64')
    df = df.dropna()
    return
