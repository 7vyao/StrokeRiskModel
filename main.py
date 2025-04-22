import warnings

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from data_processing import load_and_preprocess_data, split_data
from model import learn_structure, train_and_predict
from visualization import (
    plot_bayesian_network,
    plot_auc_curve,
    evaluate_model,
    visualize_loss_and_accuracy
)
from pgmpy.models import DiscreteBayesianNetwork

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

def main():
    file_path = 'E:\PyCharm\StatisticalModeling\data\log方法卒中复发风险保留相关变量表(活到100该死死）.csv'
    df = load_and_preprocess_data(file_path)
    print("\n预处理后数据示例：")
    print(df.head(3))
    print("\n特征分布：")
    print(df.describe())

    train_data, test_data = split_data(df)
    edges = learn_structure(train_data.copy())
    y_train, y_train_pred_prob, y_test, y_test_pred_prob = train_and_predict(train_data, test_data, edges)

    evaluate_model(y_test, y_test_pred_prob)
    viz_model = DiscreteBayesianNetwork(edges)
    viz_model.add_nodes_from(train_data.columns)
    plot_bayesian_network(viz_model)
    plot_auc_curve(y_test, y_test_pred_prob)

    visualize_loss_and_accuracy(y_train, y_train_pred_prob, y_test, y_test_pred_prob)

    print("\n评估结果：")
    print(classification_report(y_test, (np.array(y_test_pred_prob) > 0.5)))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_test_pred_prob):.3f}")


if __name__ == "__main__":
    main()
