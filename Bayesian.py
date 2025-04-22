import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import PC, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
from tabulate import tabulate
from pgmpy.estimators import BayesianEstimator
import chardet

# 忽略部分警告
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

CONFIG = {
    'bins': {
        '年龄': [0, 60, 70, 80, 100],
        'Rankin量表评分': [0, 2, 4, 6],
        'ESSEN卒中风险评分': 5  # 最大分值限制
    },
    'manual_edges': [
        ('ESSEN卒中风险评分', 'outcome'),
        ('年龄分层', 'outcome'),
        ('Rankin分层', 'outcome'),
        ('心力衰竭', 'outcome'),
        ('血脂异常', 'outcome')
    ]
}

def load_and_preprocess_data(file_path):

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    df = pd.read_csv(file_path, encoding=encoding)

    # 字段筛选
    keep_cols = [
        '年龄', '性别', '心力衰竭', 'ESSEN卒中风险评分',
        '血脂异常', '高密度脂蛋白胆固醇', '结局是否复发', 'Rankin量表评分'
    ]
    df = df[keep_cols]

    # 分类型缺失值处理
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col == '高密度脂蛋白胆固醇':
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # ：临床指标分箱
    df['年龄分层'] = pd.cut(
        df['年龄'],
        bins=CONFIG['bins']['年龄'],
        labels=[0, 1, 2, 3]
    ).astype('int64')

    df['Rankin分层'] = pd.cut(
        df['Rankin量表评分'],
        bins=CONFIG['bins']['Rankin量表评分'],
        labels=[0, 1, 2]
    ).astype('int64')

    # ESSEN评分截断
    df['ESSEN卒中风险评分'] = df['ESSEN卒中风险评分'].apply(
        lambda x: min(x, CONFIG['bins']['ESSEN卒中风险评分'])
    )

    # 编码转换
    mapping = {
        '性别': {'男': 1, '女': 0},
        '心力衰竭': {True: 1, False: 0},
        '血脂异常': {True: 1, False: 0}
    }
    df_encoded = df.replace(mapping)

    # 目标变量处理
    df_encoded['outcome'] = df_encoded['结局是否复发'].map({False: 0, True: 1})
    df_encoded = df_encoded.drop(['年龄', 'Rankin量表评分', '结局是否复发'], axis=1)

    return df_encoded


def split_data(df_encoded):
    train_data, test_data = train_test_split(
        df_encoded,
        test_size=0.3,
        stratify=df_encoded['outcome'],
        random_state=42
    )
    return train_data, test_data


# def learn_structure(df):
#
#     est = PC(data=df)
#     learned_model = est.estimate(
#         significance_level=0.01,
#         variant='parallel',
#         max_cond_vars=4,  # 根据新特征数量调整
#         ci_test='pearsonr'
#     )
#
#     merged_dag = BayesianNetwork()
#     merged_dag.add_nodes_from(df.columns)
#     merged_dag.add_edges_from(CONFIG['manual_edges'])
#
#     # 动态添加学习边
#     valid_edges = []
#     for edge in learned_model.edges():
#         temp_dag = merged_dag.copy()
#         temp_dag.add_edge(*edge)
#         if nx.is_directed_acyclic_graph(temp_dag):
#             merged_dag.add_edge(*edge)
#             valid_edges.append(edge)
#
#     print(f"最终网络结构包含 {len(merged_dag.edges())} 条边")
#     return merged_dag.edges()
def learn_structure(df):
    est = PC(data=df)
    learned_model = est.estimate(
        significance_level=0.01,
        variant='parallel',
        max_cond_vars=4,
        ci_test='pearsonr'
    )

    merged_dag = DiscreteBayesianNetwork()
    merged_dag.add_nodes_from(df.columns)
    merged_dag.add_edges_from(CONFIG['manual_edges'])

    valid_edges = []
    for edge in learned_model.edges():
        if edge[0] == 'outcome':
            continue
        temp_dag = merged_dag.copy()
        try:
            temp_dag.add_edge(*edge)
            if nx.is_directed_acyclic_graph(temp_dag):
                merged_dag.add_edge(*edge)
                valid_edges.append(edge)
        except ValueError as e:
            print(f"跳过边 {edge}，原因：{e}")

    print(f"最终网络结构包含 {len(merged_dag.edges())} 条边")
    print("有效学习边:", valid_edges)
    print("最终网络结构:", merged_dag.edges())
    return merged_dag.edges()


def train_and_predict(train_data, test_data, edges):

    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(train_data.columns)

    print("\n=== 模型结构诊断 ===")
    print("包含节点:", sorted(model.nodes()))
    print("有效边:", sorted(model.edges(), key=lambda x: x[1]))
    print("各节点父节点数:")
    for node in model.nodes():
        print(f"  {node}: {len(model.get_parents(node))}个父节点")

    # model.fit(
    #     data=train_data,
    #     estimator=MaximumLikelihoodEstimator
    # )
    try:
        model.fit(
            data=train_data,
            estimator=MaximumLikelihoodEstimator
        )
    except Exception as e:
        print(f"\n最大似然估计失败：{str(e)}")
        print("尝试使用贝叶斯估计器...")
        model.fit(
            data=train_data,
            estimator=BayesianEstimator,
            prior_type='BDeu',
            equivalent_samples_size=3,
            complete_samples_only=False
        )

    infer = VariableElimination(model)
    X_train = train_data.drop('outcome', axis=1)
    X_test = test_data.drop('outcome', axis=1)
    y_train_pred_prob = []
    y_pred_prob = []

    for _, case in X_train.iterrows():
        evidence = {col: int(case[col]) for col in X_train.columns}
        try:
            prob = infer.query(['outcome'], evidence=evidence).values[1]
            y_train_pred_prob.append(prob)
        except:
            y_train_pred_prob.append(0.5)

    for _, case in X_test.iterrows():
        evidence = {col: int(case[col]) for col in X_test.columns}
        try:
            prob = infer.query(['outcome'], evidence=evidence).values[1]
            y_pred_prob.append(prob)
        except:
            y_pred_prob.append(0.5)

    return test_data['outcome'], y_pred_prob

def evaluate_model(y_test, y_pred_prob):
    y_pred = np.array(y_pred_prob) > 0.5

    # 生成分类报告表格
    print("\n\033[1m" + "=" * 40 + "\033[0m")
    print("\033[1m分类性能报告\033[0m".center(50))
    print("\033[1m" + "=" * 40 + "\033[0m")
    print(classification_report(y_test, y_pred,
                                target_names=['未复发', '复发'],
                                digits=4))

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['预测未复发', '预测复发'],
                yticklabels=['实际未复发', '实际复发'])
    plt.xlabel('预测结果', fontsize=12, labelpad=15)
    plt.ylabel('真实结果', fontsize=12, labelpad=15)
    plt.title('中风复发预测混淆矩阵', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    # 生成指标汇总表格
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred_prob)

    metrics_table = [
        ["准确率 (Accuracy)", f"{accuracy:.4f}"],
        ["精确率 (Precision)", f"{precision:.4f}"],
        ["召回率 (Recall)", f"{recall:.4f}"],
        ["F1 分数", f"{f1:.4f}"],
        ["AUC-ROC", f"{auc:.4f}"]
    ]

    print("\n\033[1m核心指标汇总\033[0m")
    print(tabulate(metrics_table,
                   headers=["评估指标", "值"],
                   tablefmt="fancy_grid",
                   stralign="center"))

def visualize_loss_and_accuracy(y_test, y_pred_prob):
    y_pred = np.array(y_pred_prob) > 0.5
    losses = []
    accuracies = []

    for i in range(len(y_test)):
        prob = y_pred_prob[i]
        true_label = y_test.iloc[i]
        # 添加小的epsilon值，避免log(0)
        loss = -np.log(prob + 1e-10) if true_label == 1 else -np.log(1 - prob + 1e-10)
        losses.append(loss)
        accuracies.append(int(y_pred[i] == true_label))

    # 使用样本序号代替epoch
    samples = range(1, len(losses) + 1)

    plt.figure(figsize=(14, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'r-', alpha=0.7)
    plt.title('Sample Prediction Loss', fontsize=12, pad=15)
    plt.xlabel('Test Sample Index', fontsize=10)
    plt.ylabel('Cross-Entropy Loss', fontsize=10)
    plt.grid(linestyle='--', alpha=0.5)

    # 准确率平滑处理
    smoothed_accuracies = np.cumsum(accuracies) / (np.arange(len(accuracies)) + 1)

    # 准确率分布
    plt.subplot(1, 2, 2)
    plt.plot(smoothed_accuracies, 'g-', alpha=0.7)
    plt.title('Sample Prediction Accuracy', fontsize=12, pad=15)
    plt.xlabel('Test Sample Index', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()


def main():
    file_path = 'E:\PyCharm\StatisticalModeling\data\log方法卒中复发风险保留相关变量表(活到100该死死）.csv'
    df = load_and_preprocess_data(file_path)
    print("\n预处理后数据示例：")
    print(df.head(3))
    print("\n特征分布：")
    print(df.describe())

    train_data, test_data = split_data(df)
    edges = learn_structure(train_data.copy())
    y_test, y_pred_prob = train_and_predict(train_data, test_data, edges)

    evaluate_model(y_test, y_pred_prob)
    visualize_loss_and_accuracy(y_test, y_pred_prob)

    print("\n评估结果：")
    print(classification_report(y_test, (np.array(y_pred_prob) > 0.5)))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.3f}")
    visualize_loss_and_accuracy(y_test, y_pred_prob)


if __name__ == "__main__":
    main()
