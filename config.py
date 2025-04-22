CONFIG = {
    'bins': {
        '年龄': [0, 60, 100],
        'Rankin量表评分': [0, 2, 4, 6],
        'ESSEN卒中风险评分': 6
    },
    'manual_edges': [
        ('ESSEN卒中风险评分', 'outcome'),
        ('年龄分层', 'outcome'),
        ('Rankin分层', 'outcome'),
        ('心力衰竭', 'outcome'),
        ('血脂异常', 'outcome')
    ]
}
