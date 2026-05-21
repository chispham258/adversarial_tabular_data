import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from art.attacks.evasion import ZooAttack, HopSkipJump, LowProFool, AutoProjectedGradientDescent
from art.estimators.classification import XGBoostClassifier, SklearnClassifier, PyTorchClassifier

from PermuteAttack.src.ga_attack import GA_Counterfactual, GAdvExample, alibi_ord_to_ohe, alibi_ohe_to_ord
from PermuteAttack.src.utils import plot_graph, create_onehot_map


def prepare_data(df):

    train = df[df["is_train"] == 1]
    test = df[df["is_train"] == 0]

    Y_train = train["target"].values
    X_train = train.drop(columns=["name", "prediction", "is_train", "target"]).to_numpy()

    Y_test = test["target"].values
    X_test = test.drop(columns=["name", "prediction", "is_train", "target"]).to_numpy()

    X_columns = train.drop(columns=["name", "prediction", "is_train", "target"]).columns

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    Y_train_one = np.zeros((Y_train.size, Y_train.max() + 1))
    Y_train_one[np.arange(Y_train.size), Y_train] = 1

    Y_test_one = np.zeros((Y_test.size, Y_test.max() + 1))
    Y_test_one[np.arange(Y_test.size), Y_test] = 1

    return X_train, X_test, Y_train, Y_test, Y_train_one, Y_test_one, list(X_columns)


def train_sklearn_model(X_train, X_test, Y_train, Y_test, Y_train_one, Y_test_one, selected_model, dataset_name):

    if selected_model == "lin":
        model = LogisticRegression()
    elif selected_model == "svm":
        model = SVC(probability=True)
    elif selected_model == "xgb":
        model = GradientBoostingClassifier(n_estimators=500)

    classifier = SklearnClassifier(model=model)
    classifier.fit(X_train, Y_train_one)
    model.fit(X_train, Y_train)

    predictions = classifier.predict(X_test)
    print(f"BACC: {balanced_accuracy_score(Y_test, np.argmax(predictions, axis=1))}")
    bacc_dict = {'bacc_test': balanced_accuracy_score(Y_test, np.argmax(predictions, axis=1))}
    pd.DataFrame(bacc_dict, index=[0]).to_csv(os.path.join(path_bacc, "bacc_" + selected_model + "_org_" + dataset_name + ".csv"), index=False)

    # save original data
    pred_train = classifier.predict(X_train)
    pred_train_df = pd.DataFrame(pred_train)
    pred_train_df.columns = [f"score_{i}" for i in range(pred_train_df.shape[1])]
    train_df = pd.DataFrame(X_train)
    train_df["name"] = list(range(train_df.shape[0]))
    train_df["is_train"] = 1
    train_df["target"] = Y_train
    train_df["prediction"] = np.argmax(pred_train, axis=1)
    train_df = pd.concat([train_df, pred_train_df], axis=1)

    test_nrows = train_df.shape[0]+1
    pred_test = classifier.predict(X_test)
    pred_test_df = pd.DataFrame(pred_test)
    pred_test_df.columns = [f"score_{i}" for i in range(pred_test_df.shape[1])]
    test_df = pd.DataFrame(X_test)
    test_df["name"] = list(range(test_nrows,test_nrows+test_df.shape[0]))
    test_df["is_train"] = 0
    test_df["target"] = Y_test
    test_df["prediction"] = np.argmax(pred_test, axis=1)
    test_df = pd.concat([test_df, pred_test_df], axis=1)

    train_test = pd.concat([train_df, test_df])
    train_test.to_csv(os.path.join(path_attacks, dataset_name + "_" + selected_model + "_org" + ".csv"), index=False)

    return classifier, model, X_test, Y_test, Y_test_one, test_nrows


def conduct_attack(classifier, X_test, Y_test, Y_test_one, test_nrows, selected_attack):

    if selected_attack == "zoo":
        attack = ZooAttack(classifier=classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=200,
                           binary_search_steps=10,
                           initial_const=1e-3, abort_early=True, use_resize=False, use_importance=False, nb_parallel=5,
                           batch_size=1, variable_h=0.01)
        x_test_adv = attack.generate(x=X_test, y=Y_test)
    elif selected_attack == "hsj":
        attack = HopSkipJump(classifier=classifier)
        x_test_adv = attack.generate(x=X_test, y=Y_test)
    elif selected_attack == "lpf":
        attack = LowProFool(classifier=classifier)
        attack_fit = attack.fit_importances(x=X_test, y=Y_test)
        x_test_adv = attack_fit.generate(x=X_test, y=Y_test_one)

    pred_test = classifier.predict(x_test_adv)
    pred_test_df = pd.DataFrame(pred_test)
    pred_test_df.columns = [f"score_{i}" for i in range(pred_test_df.shape[1])]
    test_df = pd.DataFrame(x_test_adv)
    test_df["name"] = list(range(test_nrows,test_nrows+test_df.shape[0]))
    test_df["is_train"] = 0
    test_df["target"] = Y_test
    test_df["prediction"] = np.argmax(pred_test, axis=1)
    test_df = pd.concat([test_df, pred_test_df], axis=1)
    test_df.to_csv(os.path.join(path_attacks, dataset_name + "_" + selected_model + "_" + selected_attack + ".csv"), index=False)

    print(f"BACC: {balanced_accuracy_score(test_df['target'], test_df['prediction'])}")
    bacc_dict = {'bacc_test': balanced_accuracy_score(test_df['target'], test_df['prediction'])}
    pd.DataFrame(bacc_dict, index=[0]).to_csv(os.path.join(path_bacc, "bacc_" + selected_model + "_" + selected_attack + "_" + dataset_name + ".csv"), index=False)


def conduct_permute_attack(model, X_test, Y_test, X_train, test_nrows, features):

    ga = GAdvExample(feature_names=features,
                     sol_per_pop=30, num_parents_mating=10, cat_vars_ohe=None,
                     num_generations=100, n_runs=5, black_list=[],
                     verbose=False, beta=.95)

    x_test_adv = []

    for idx_test in tqdm(range(X_test.shape[0])):
        x_all, x_changes, x_success = ga.attack(model, x=X_test[idx_test,:],x_train=X_train)
        x_test_adv.append(x_success[0]) if len(x_success) > 0 else x_test_adv.append(X_test[idx_test,:])

    pred_test = model.predict_proba(x_test_adv)
    pred_test_df = pd.DataFrame(pred_test)
    pred_test_df.columns = [f"score_{i}" for i in range(pred_test_df.shape[1])]
    test_df = pd.DataFrame(x_test_adv)
    test_df["name"] = list(range(test_nrows,test_nrows+test_df.shape[0]))
    test_df["is_train"] = 0
    test_df["target"] = Y_test
    test_df["prediction"] = np.argmax(pred_test, axis=1)
    test_df = pd.concat([test_df, pred_test_df], axis=1)
    test_df.to_csv(os.path.join(path_attacks, dataset_name + "_" + selected_model + "_" + selected_attack + ".csv"), index=False)

    print(f"BACC: {balanced_accuracy_score(test_df['target'], test_df['prediction'])}")
    bacc_dict = {'bacc_test': balanced_accuracy_score(test_df['target'], test_df['prediction'])}
    pd.DataFrame(bacc_dict, index=[0]).to_csv(os.path.join(path_bacc, "bacc_" + selected_model + "_" + selected_attack + "_" + dataset_name + ".csv"), index=False)

directory = "data"
list_of_files = os.listdir(directory)

for input_file in list_of_files:

    # input_file = list_of_files[0]
    filename = input_file.split("/")[-1]
    dataset_name = filename.split(".")[0]

    output_path = "results"
    path_attacks = os.path.join(output_path, dataset_name, "attacks")
    path_bacc = os.path.join(output_path, dataset_name, "bacc")
    os.makedirs(path_attacks, exist_ok=True)
    os.makedirs(path_bacc, exist_ok=True)

    df = pd.read_csv(os.path.join(directory, input_file))

    models = ["svm", "lin", "xgb"]
    attacks = ["per", "zoo", "hsj", "lpf"]

    X_train, X_test, Y_train, Y_test, Y_train_one, Y_test_one, X_columns = prepare_data(df)

    for selected_model in models:

        print(selected_model)

        classifier, model, X_test, Y_test, Y_test_one, test_nrows = train_sklearn_model(X_train, X_test, Y_train, Y_test, Y_train_one, Y_test_one,
                                                                                 selected_model, dataset_name)

        for selected_attack in attacks:

            print(selected_attack)

            try:
                if selected_attack == "per":
                    conduct_permute_attack(model, X_test, Y_test, X_train, test_nrows, X_columns)
                else:
                    conduct_attack(classifier, X_test, Y_test, Y_test_one, test_nrows, selected_attack)
            except:
                print("It is impossible to conduct this attack")
