import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import PC, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
from config import CONFIG

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

    try:
        # 使用贝叶斯估计器，并增加等效样本大小
        model.fit(
            data=train_data,
            estimator=BayesianEstimator,
            prior_type='BDeu',
            equivalent_samples_size=150,
            complete_samples_only=False
        )
    except Exception as e:
        print(f"\n参数估计失败：{str(e)}")
        # 如果仍然失败，可尝试调整其他参数或使用其他估计器
        model.fit(
            data=train_data,
            estimator=MaximumLikelihoodEstimator
        )

    infer = VariableElimination(model)
    X_train = train_data.drop('outcome', axis=1)
    X_test = test_data.drop('outcome', axis=1)
    y_train_pred_prob = []
    y_test_pred_prob = []

    # 增加异常处理，确保预测过程的稳定性
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
            y_test_pred_prob.append(prob)
        except:
            y_test_pred_prob.append(0.5)

    return (train_data['outcome'], y_train_pred_prob,
            test_data['outcome'], y_test_pred_prob)
