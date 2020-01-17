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
    df2 = df.drop_duplicates(subset=['patient_nbr'], keep='first')

    df = df.drop(["weight", "payer_code", "medical_specialty"], axis=1)
    df = df.drop(df[df.gender == 'Unknown/Invalid'].index)
    meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
            'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    df = df.drop(meds, axis=1)

    df['readmitted'] = df['readmitted'].replace('>30', 0)
    df['readmitted'] = df['readmitted'].replace('<30', 1)
    df['readmitted'] = df['readmitted'].replace('NO', 0)
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
                "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
    df['age'] = df.age.map(age_dict)
    df['age'] = df['age'].astype('int64')
    return df