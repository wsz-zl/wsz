'''环境准备'''
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split, StratifiedKFold   #数据划分和交叉验证
from sklearn.ensemble import RandomForestClassifier  #随机森林分类器
from sklearn import metrics  #评估指标（F1）
import joblib                #模型和特征的缓存保存与加载
from concurrent.futures import ProcessPoolExecutor  #多进程并行处理
import os   #文件路径操作
import gc   #垃圾回收控制
import multiprocessing  #多进程支持

'''数据加载'''
dataset = pd.read_csv(r"C:\Users\WSZ\Desktop\Train.csv")      #加载训练数据
mirna_seqdf = pd.read_csv(r"C:\Users\WSZ\Desktop\mirna_seq.csv")[['mirna', 'seq']]    #加载mirna_seq数据
gene_seqdf = pd.read_csv(r"C:\Users\WSZ\Desktop\gene_seq.csv")[['label', 'sequence']]  #加载gene_seq数据

'''数据预处理'''
#清理mirna_seq和gene_seq中的换行符
mirna_seqdf['seq'] = mirna_seqdf['seq'].str.replace('\n', '')
gene_seqdf['sequence'] = gene_seqdf['sequence'].str.replace('\n', '')

#创建序列映射字典提高查询效率
mirna_seq_dict = dict(zip(mirna_seqdf['mirna'], mirna_seqdf['seq']))    #mirna名称到序列的映射
gene_seq_dict = dict(zip(gene_seqdf['label'], gene_seqdf['sequence']))  #gene名称到序列的映射

#数据拆分,提取特征所需的列
dataset_mirna = dataset['miRNA']
dataset_gene = dataset['gene']
dataset_label = dataset['label']

#特征提取函数
def calculate_gc_content(seq):
    #计算GC碱基占比
    gc_count = seq.count('G') + seq.count('C') + seq.count('g') + seq.count('c')  #统计GC碱基数量
    return gc_count / len(seq) if len(seq) > 0 else 0   #返回比例，避免除零错误

def calculate_complementarity(mirna_seq, gene_seq):
    #计算miRNA与gene序列的互补性
    seed = mirna_seq[1:7] if len(mirna_seq) >= 7 else mirna_seq  #取miRNA的种子区域
    # 在gene序列中查找最佳互补匹配
    max_complementarity = 0
    for i in range(len(gene_seq) - len(seed) + 1):
        window = gene_seq[i:i + len(seed)]
        #统计互补碱基对数量（A-T/U, C-G）
        complementarity = sum(1 for a, b in zip(seed, window)
                              if (a == 'A' and b == 'T') or (a == 'T' and b == 'A') or
                              (a == 'C' and b == 'G') or (a == 'G' and b == 'C') or
                              (a == 'A' and b == 'U') or (a == 'U' and b == 'A'))
        max_complementarity = max(max_complementarity, complementarity)  #更新最大值
    return max_complementarity / len(seed) if len(seed) > 0 else 0  #归一化到[0,1]

def kmer_features(n, seq, alphabet):
    #生成k-mer特征向量
    kmer_keys = [''.join(p) for p in itertools.product(alphabet, repeat=n)]   #所有可能的k-mer组合
    kmer_dict = {k: 0 for k in kmer_keys}  #初始化计数字典
    #序列长度不足时返回全0
    if len(seq) < n:
        return kmer_dict
    #统计k-mer
    for i in range(len(seq) - n + 1):
        kmer = seq[i:i + n]
        if kmer in kmer_dict:
            kmer_dict[kmer] += 1

    # 归一化处理
    total = sum(kmer_dict.values())
    if total > 0:
        for k in kmer_dict:
            kmer_dict[k] = kmer_dict[k] / total

    return kmer_dict

def extract_features(row):
    #为单个样本提取所有特征
    mirna_name, gene_name = row           #解包当前样本的mirna和gene名称
    try:
        mirna_seq = mirna_seq_dict.get(mirna_name, '')   # 获取mirna序列
        gene_seq = gene_seq_dict.get(gene_name, '')       # 获取gene序列
        #如果任一序列缺失，返回None
        if not mirna_seq or not gene_seq:
            return None

        # 基础特征（mirna、gene长度；mirna、gene的GC含量）
        features = {
            'mirna_len': len(mirna_seq),
            'gene_len': len(gene_seq),
            'mirna_gc': calculate_gc_content(mirna_seq),
            'gene_gc': calculate_gc_content(gene_seq),
            'complementarity': calculate_complementarity(mirna_seq, gene_seq)   #互补性
        }
        # 3-mer特征
        features.update(kmer_features(3, mirna_seq, 'UCGA'))    #mirna的3-mer
        features.update(kmer_features(3, gene_seq, 'TCGA'))     #基因的3-mer
        # 4-mer特征
        features.update(kmer_features(4, mirna_seq, 'UCGA'))
        features.update(kmer_features(4, gene_seq, 'TCGA'))

        return features
    # 捕获异常并打印错误信息
    except Exception as e:
        print(f"Error processing {mirna_name}-{gene_name}: {e}")
        return None

#并行特征提取
def parallel_feature_extraction(mirna_list, gene_list, cache_path="feature_cache.pkl"):
    #并行提取特征，使用缓存加速
    #检查缓存是否存在
    #存在
    if os.path.exists(cache_path):
        print("Loading features from cache...")  #打印提示
        return joblib.load(cache_path)    #直接加载缓存的特征DataFrame
    #不存在
    print("Extracting features...") #打印提示
    features_list = []  #存储所有样本的特征字典
    total = len(mirna_list)   #总样本数

    #使用并行处理
    if multiprocessing.current_process().name == 'MainProcess':
        #主进程中使用并行
        #创建进程池
        with ProcessPoolExecutor() as executor:
            futures = []  #存储future对象
            #遍历所有样本
            for i in range(total):
                futures.append(executor.submit(extract_features, (mirna_list[i], gene_list[i])))
            #遍历future对象
            for i, future in enumerate(futures):
                features = future.result() #获取结果
                #如果特征有效
                if features is not None:
                    features_list.append(features)   #添加到列表
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{total} samples...")
                    gc.collect()
    else:
        # 子进程中使用串行
        print("Running in child process, using single process mode")
        for i in range(total):   #串行处理
            features = extract_features((mirna_list[i], gene_list[i]))  #提取特征
            if features is not None:
                features_list.append(features)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total} samples...")

    # 转换为DataFrame并保存缓存
    feature_df = pd.DataFrame(features_list)  #将特征字典列表转为DataFrame
    joblib.dump(feature_df, cache_path)      #保存为缓存文件
    print(f"Features saved to cache: {cache_path}")
    return feature_df       #返回特征DataFrame

#标签预处理
Y = np.array([1 if lbl == 'Functional MTI' else 0 for lbl in dataset_label]) #将标签转换为二进制数组

#模型训练与交叉验证
def train_model_with_cv(X, y):
    #使用5折交叉验证训练随机森林模型，并选择最佳F1分数的模型
    print("Starting cross-validation training...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   #分层10折交叉验证
    best_model = None   #最佳模型
    best_f1 = 0         #最佳F1分数
    models = []         #存储所有折叠的模型
    # 交叉验证循环，遍历每一折
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]    #划分训练集和验证集
        y_train, y_val = y[train_idx], y[val_idx]

        # 处理类别不平衡
        class_weights = {0: 1.0, 1: len(y_train) / (2 * np.sum(y_train))}   #正样本权重更高
        # 初始化随机森林模型
        model = RandomForestClassifier(
            n_estimators=200,    #树的数量
            max_depth=20,        #最大深度
            min_samples_split=5,   #分裂所需最小样本数
            min_samples_leaf=2,    #叶节点最小样本数
            class_weight=class_weights,  #类别权重
            n_jobs=-1,                   #使用所有CPU核心
            random_state=42 + fold       #随机种子
        )

        model.fit(X_train, y_train)   #训练模型
        models.append(model)          #保存模型

        # 验证集评估
        y_pred = model.predict(X_val)        #预测验证集
        f1 = metrics.f1_score(y_val, y_pred)     #计算F1分数
        acc = metrics.accuracy_score(y_val, y_pred)    #计算准确率
        print(f"Fold {fold + 1}: F1={f1:.4f}, Acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1   #更新最佳F1
            best_model = model   #更新最佳模型

    # 保存最佳模型
    os.makedirs("./model", exist_ok=True)
    joblib.dump(best_model, "./model/best_model.m")
    print(f"Best model saved with F1: {best_f1:.4f}")

    return best_model, models

#主流程
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows多进程支持
    # 特征提取
    X = parallel_feature_extraction(dataset_mirna.tolist(), dataset_gene.tolist(), "train_features.pkl")
    # 移除可能的空值
    valid_idx = X.notnull().all(axis=1)       #检查是否有空值
    X = X[valid_idx]     #过滤有效样本
    y = Y[valid_idx]     #同步过滤标签

    # 训练模型
    best_model, all_models = train_model_with_cv(X, y)
    # 在完整数据集上训练最终模型
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: len(y) / (2 * np.sum(y))},
        n_jobs=-1,
        random_state=42
    )
    final_model.fit(X, y)     #训练最终模型
    joblib.dump(final_model, "./model/final_model.m")

#预测与提交
def generate_submission(test_path=r"C:\Users\WSZ\Desktop\test_dataset.csv"):
    #生成预测结果
    test_data = pd.read_csv(test_path)   #加载测试数据
    predict_mirna = test_data['miRNA'].tolist()  # 提取mirna列表
    predict_gene = test_data['gene'].tolist()    # 提取gene列表

    # 提取测试集特征
    X_test = parallel_feature_extraction(predict_mirna, predict_gene, "test_features.pkl")

    # 加载模型预测（使用完整模型）
    final_model = joblib.load("./model/final_model.m")
    predictions = final_model.predict(X_test)       #预测测试集

    # 生成提交文件
    test_data['results'] = predictions         #添加预测结果列
    submission = test_data[['miRNA', 'gene', 'results']]    #选择需要的列
    submission.columns = ['miRNA', 'gene', 'results']    # 确保列名正确
    submission.to_csv('submission.csv', index=False)      #保存为CSV
    print("Submission file generated: submission.csv")
    return submission      #返回提交数据

# 执行预测
if __name__ == "__main__":
    generate_submission()       #执行预测并生成提交文件
