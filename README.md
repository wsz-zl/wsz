# wsz
机器学习
数据说明
数据文件夹包含4个文件，依次为：
文件类别	文件名	文件内容
训练集	train.csv	关系对训练数据集，标签Functional MTI，Non-Functional MTI
测试集	test.csv	Test_without_label.csv 无标签
基因序列文件	gene_seq.csv	Gene的相关序列的文件，提供名字和序列
microRNA序列文件	mirna_seq.csv	miRNA的相关序列文件，提供名字和序列
提交样例	submission.csv	三个字段，miRNA，gene以及预测的结果results
赛者以csv文件格式提交，提交模型结果到大数据竞赛平台，平台进行在线评分，实时排名。目前平台仅支持单文件提交，即所有提交内容需要放在一个文件中；submission.csv文件字段如下：
