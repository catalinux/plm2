from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive',
          'tab:cyan', 'tab:gray']
MARKERS = ['o', 'v', 's', '<', '>', '8', '^', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']


def report_missing_values(df):
    attr_list = []
    missing_value_list = []
    for attr in attributes:
        missing_count = df[attr].isnull().sum()
        if missing_count > 0:
            attr_list.append(attr)
            missing_value_list.append(round(missing_count / len(df[attr]) * 100))

    fig, ax = plt.subplots()
    y_pos = np.arange(len(attr_list))
    plt.bar(y_pos, missing_value_list)
    plt.xticks(y_pos, attr_list)
    plt.ylabel('Percentage')

    plt.show()


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


def plot3d(X, y_pred, y_true, mode=None, centroids=None, name=''):
    transformer = None
    X_r = X
    if mode is not None:
        transformer = mode(n_components=3)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 3, 'plot2d only works with 3-dimensional data'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.elev = 30
    ax.azim = 12

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
    plt.title(name)
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
    df['readmitted'] = pd.Categorical(df['readmitted']).codes
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

    df['diag_1'] = diag_mapper_icd9(df['diag_1'])
    df['diag_2'] = diag_mapper_icd9(df['diag_2'])
    df['diag_3'] = diag_mapper_icd9(df['diag_3'])

    # construct dictionaries for the mapping
    o = 'other'
    e = 'dead'
    h = 'hospice'
    c = 'child_newborn'
    dh = 'discharged to health facility'
    transfer = 'transfer'
    s_null = 'lim info'
    ref = 'other referral'
    d_map = {1: 'Home',
             2: dh,  # only 1 3 and 6 don't need to be mapped, though could be anyway
             3: 'Dis to SNF',
             4: dh,
             5: dh,
             6: 'Home w Home Care',
             7: 'left AMA',
             8: dh,
             9: 'inpatient',
             10: dh,
             11: e,
             12: 'pat_outpat',
             13: h,
             14: h,
             15: dh,
             16: dh,
             17: dh,
             18: s_null,
             19: e,
             20: e,
             21: e,
             22: dh,
             23: dh,
             24: dh,
             25: s_null,
             26: s_null,
             27: dh,
             28: dh,
             29: dh,
             30: dh}

    s_map = {1: 'Physician Referral',
             2: ref,  # 1 7 and 17 don't have to be mapped
             3: ref,
             4: transfer,
             5: transfer,
             6: transfer,
             7: 'Emergency room',
             8: o,
             9: s_null,
             10: transfer,
             11: o,
             12: o,
             13: c,
             14: o,
             15: s_null,
             17: s_null,
             18: transfer,
             19: o,
             20: s_null,
             21: s_null,
             22: transfer,
             23: c,
             24: c,
             25: transfer,
             26: transfer
             }
    t_map = {1: 'Emergency',
             2: 'Urgent',
             3: 'Elective',
             4: o,
             5: s_null,
             6: s_null,
             7: o,
             8: s_null}

    df['admission_source_id'] = df['admission_source_id'].map(s_map)
    df['admission_type_id'] = df['admission_type_id'].map(t_map)
    df['discharge_disposition_id'] = df['discharge_disposition_id'].map(d_map)

    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)
    s = df["admission_source_id"].value_counts()
    df['admission_source_id'].isin(s.index[s >= 30]).index
    # df = df.drop(['discharge_disposition_id', 'admission_source_id'], axis=1)
    df = df[df["service_utilization"] < 11]
    df.drop("admission_source_id", axis=1, inplace=True)
    df.dropna(inplace=True)
    drop_numerical_outliers(df, 6)

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


def diag_mapper_icd9(diag):
    """Used on the columns diag_1, diag_2 and diag_3.
    Takes a list of icd9 codes and maps them to each category defined by icd9.
    This makes it easier to process in the machine learning algorithms.

    For a list of what each category represents see:
        https://en.wikipedia.org/wiki/List_of_ICD-9_codes

    Even reducing down the the number of codes to 19 categories still means
    there's a lot of columns to deal with. For this reason a second mode of this
    function was created which maps to 5 categories, the 4 most common
    of the 19 categories and a combined one for everything else. Which of these
    maps is used is determined by the variable 'over'.

    Lists can have their values changed by the map method. Map takes a
    dictionary of values with the starting values as keys and the end values
    as dictionary values. In each part of the if statement the code tries to
    convert the categories of diag (the input column) to a float. It then looks
    through the list of categories to map to, checking whether it's between
    the current element of the list it's looking at and the next one. If it
    can't make the value into a float it checks to see if that value begins
    with E or V.
    """

    over = 0
    diag_map = {}
    c_list = diag.unique()
    if over == 0:
        icd9 = [1, 140, 240, 280, 290, 320, 360, 390, 460, 520, 580, 630, 680, 710, 740, 760,
                780, 800, 1000]

        for i in range(len(c_list)):
            try:
                b = float(c_list[i])
                for j in range(len(icd9) - 1):
                    if b >= icd9[j] and b < icd9[j + 1]:
                        diag_map[c_list[i]] = str(icd9[j])
                        break

            except ValueError:
                if c_list[i][0] == 'E':
                    diag_map[c_list[i]] = 'E'
                elif c_list[i][0] == 'V':
                    diag_map[c_list[i]] = 'V'

    elif over == 1:
        icd_cut = [[390, 460], [240, 280], [460, 520], [580, 630]]
        for i in range(len(c_list)):
            num = 0
            try:
                b = float(c_list[i])
                for j in icd_cut:
                    if b >= j[0] and b < j[1]:
                        diag_map[c_list[i]] = str(j[0])
                        num = 1
                        break

                    if num == 0:
                        diag_map[c_list[i]] = 'Other'

            except ValueError:
                diag_map[c_list[i]] = 'Other'

    diag_new = diag.map(diag_map)
    return diag_new


def prepare_data(data):
    cat_df_list = ['race',  'admission_type_id', 'discharge_disposition_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change']
    print("Encoding : ", cat_df_list)
    for i in cat_df_list:
        le = LabelEncoder()
        le.fit(list(data[i].unique()))
        data.loc[:, i] = le.transform(data[i])
    prepped_data = data
    prepped_data = pd.get_dummies(data, prefix=cat_df_list, prefix_sep='_', columns=cat_df_list, drop_first=True)
    return  prepped_data


from scipy import stats

def drop_numerical_outliers(df, z_thresh=4):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, reduce=False) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)