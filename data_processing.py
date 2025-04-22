import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from config import CONFIG

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
        labels=[0, 1]
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