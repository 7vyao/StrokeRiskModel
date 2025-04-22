import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
import networkx as nx
from sklearn.metrics import roc_curve, auc

def plot_bayesian_network(model, figsize=(10, 6)):
    """
    可视化贝叶斯网络结构，图形更加紧凑
    """
    plt.figure(figsize=figsize)

    # 创建有向图
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # 设置节点布局
    pos = nx.spring_layout(G, k=0.6, seed=42)  # k值较小，节点之间的间距减少

    # 绘制节点
    node_colors = ['#FF6B6B' if node == 'outcome' else '#4ECDC4' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5000)

    # 绘制边
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#555555',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        node_size=5000
    )

    # 绘制标签
    nx.draw_networkx_labels(
        G, pos,
        font_family='SimHei',  # 支持中文
        font_size=12,
        font_color='white'
    )

    # 设置标题和图例
    plt.title("Bayesian network structure of stroke recurrence risk", fontsize=14, pad=20)
    plt.axis('off')

    # 添加图例说明
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Ending Node', markerfacecolor='#FF6B6B', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Feature Node', markerfacecolor='#4ECDC4', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_auc_curve(y_test, y_test_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机猜测的对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
                xticklabels=['Predicted no recurrence', 'Predicting recurrence'],
                yticklabels=['No actual recurrence', 'Actual relapse'])
    plt.xlabel('Prediction results', fontsize=12, labelpad=15)
    plt.ylabel('Real results', fontsize=12, labelpad=15)
    plt.title('Confusion matrix for stroke recurrence prediction', fontsize=14, pad=20)
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


def visualize_loss_and_accuracy(y_train, y_train_pred_prob,
                                y_test, y_test_pred_prob):
    # 同时处理两组数据
    datasets = {
        'Train': (y_train, y_train_pred_prob),
        'Test': (y_test, y_test_pred_prob)
    }

    plt.figure(figsize=(14, 6))

    # === 改进后的损失曲线对比 ===
    plt.subplot(1, 2, 1)
    for name, (y_true, y_prob) in datasets.items():
        losses = [
            -np.log(prob + 1e-10) if true == 1 else -np.log(1 - prob + 1e-10)
            for prob, true in zip(y_prob, y_true)
        ]
        # 平滑处理损失曲线，增大窗口大小以获得更平滑的曲线
        window_size = 25  # 增大窗口大小
        smoothed_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_loss, alpha=0.7, label=f'{name} Loss')
    plt.title('Smoothed Loss Curve Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()

    # === 改进后的准确率曲线对比 ===
    plt.subplot(1, 2, 2)
    for name, (y_true, y_prob) in datasets.items():
        y_pred = np.array(y_prob) > 0.5
        smoothed_acc = np.cumsum(y_pred == y_true) / (np.arange(len(y_true)) + 1)
        # 平滑处理准确率曲线，增大窗口大小以获得更平滑的曲线
        window_size = 25  # 增大窗口大小
        smoothed_acc = np.convolve(smoothed_acc, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_acc, alpha=0.7, label=f'{name} Accuracy')
    plt.title('Smoothed Accuracy Trend Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Smoothed Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()