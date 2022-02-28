import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, auc, roc_curve, average_precision_score
argparser = argparse.ArgumentParser()
argparser.add_argument("--training_path", dest="training_path")
argparser.add_argument("--training_label_path", dest="training_label_path")
argparser.add_argument("--test_path", dest="test_path")
argparser.add_argument("--test_label_path", dest="test_label_path")
argparser.add_argument("--output_path", dest="output_path")
argparser.add_argument("--n_trees", dest="n_trees", type=int, default=1000)
argparser.add_argument("--k_fold", dest="k_fold", type=int, default=10)


def get_xy(known_abundance_dict, label_dict, selected=None):
    assert known_abundance_dict.keys() == label_dict.keys()
    if selected is not None:
        selected_to_label = {l: i for i, l in enumerate(sorted(selected))}
    X, y = [], []
    sample_id_list = []
    id_list = sorted(known_abundance_dict.keys())
    for sample_id in id_list:
        if selected is not None:
            if label_dict[sample_id] in selected:
                sample_id_list.append(sample_id)
                X.append(known_abundance_dict[sample_id])
                y.append(selected_to_label[label_dict[sample_id]])
        else:
            sample_id_list.append(sample_id)
            X.append(known_abundance_dict[sample_id])
            y.append(label_dict[sample_id])
    return np.array(X), np.array(y), sample_id_list

def get_data_from_csv(data_path, label_path):
    t = pd.read_csv(data_path, index_col=0)
    lt = pd.read_csv(label_path, index_col=0).transpose()
    label_dict = {k: lt[k].disease for k in lt.keys()}
    data_dict = {k: v for k, v in t.to_dict("list").items() if k in label_dict}
    X, y, _ = get_xy(data_dict, label_dict)
    return X, y

def get_label_dict(label_path):
    lt = pd.read_csv(label_path, index_col=0).transpose()
    label_dict = {k: lt[k].disease for k in lt.keys()}
    return label_dict

def get_train_test_from_csv(training_path, training_label_path, test_path, test_label_path):
    training_t = pd.read_csv(training_path, index_col=0)
    training_label_dict = get_label_dict(training_label_path)
    training_t = training_t[list(training_label_dict.keys())]
    test_t = pd.read_csv(test_path, index_col=0)
    test_label_dict = get_label_dict(test_label_path)
    test_t = test_t[list(test_label_dict.keys())]
    intersection_index = training_t.index.intersection(test_t.index)
    training_t = training_t.loc[intersection_index]
    test_t = test_t.loc[intersection_index]
    x_train, y_train, _ = get_xy(training_t.to_dict("list"), training_label_dict)
    x_test, y_test, _ = get_xy(test_t.to_dict("list"), test_label_dict)
    return x_train, x_test, y_train, y_test

def train_test(training_path, training_label_path, test_path, test_label_path, n_iters=20, n_estimators=1000):
    x_train, x_test, y_train, y_test = get_train_test_from_csv(training_path, training_label_path, test_path, test_label_path)
    acc_list = []
    roc_list = []
    for _ in range(n_iters):
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(x_train, y_train)
        y_prob = clf.predict_proba(x_test)
        y_pred = clf.predict(x_test)
        acc_list.append(accuracy_score(y_test, y_pred))
        roc_list.append(roc_auc_score(y_test, y_prob[:, 1]))
    print("Average accuracy: %.4f\t Standard deviation: %.4f" % (np.mean(acc_list), np.std(acc_list)))
    print("Average auroc: %.4f\t Standard deviation: %.4f" % (np.mean(roc_list), np.std(roc_list)))
    return acc_list, roc_list

def run_cv(data_path, label_path, n_iters=20, n_estimators=1000, k_fold=10):
    x, y = get_data_from_csv(data_path, label_path)
    acc_list = []
    roc_list = []
    for _ in range(n_iters):
        kf = KFold(n_splits=k_fold, shuffle=True)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(n_estimators=n_estimators)
            clf.fit(x_train, y_train)
            y_prob = clf.predict_proba(x_test)
            y_pred = clf.predict(x_test)
            acc_list.append(accuracy_score(y_test, y_pred))
            roc_list.append(roc_auc_score(y_test, y_prob[:, 1]))
    print("Average accuracy: %.4f\t Standard deviation: %.4f" % (np.mean(acc_list), np.std(acc_list)))
    print("Average auroc: %.4f\t Standard deviation: %.4f" % (np.mean(roc_list), np.std(roc_list)))
    return acc_list, roc_list

def main():
    args = argparser.parse_args()
    n_estimators = args.n_trees
    if args.training_path != args.test_path:
        acc_list, roc_list = train_test(args.training_path, args.training_label_path, args.test_path, args.test_label_path, n_estimators=n_estimators)
    else:
        acc_list, roc_list = run_cv(args.training_path, args.training_label_path, n_estimators=n_estimators, k_fold=args.k_fold)
    with open(args.output_path, 'w') as f:
        f.write("%.4f,%.4f\n" % (np.mean(acc_list), np.std(acc_list)))
        f.write("%.4f,%.4f" % (np.mean(roc_list), np.std(roc_list)))

if __name__ == "__main__":
    main()
