from sklearn import metrics
import pandas as pd
def performance_evaluation(ground_truth, results):
    gt = []
    pred = []
    for i in range(min(len(ground_truth), len(results))):
        gt.append(ground_truth['anomaly'][i])
        pred.append(results['action'][i])
    acc = metrics.accuracy_score(gt, pred)
    precision, recall, F1, _ = metrics.precision_recall_fscore_support(gt, pred, average='binary')
    print(metrics.confusion_matrix(gt, pred))

    print('acc:', acc, 'precision:', precision, 'recall:', recall, 'F1 score:', F1)
    return 0
ground_truth = pd.read_csv('traffic_data/synthetic_data.csv', header=None, usecols=[0,1,2], names= ['id', 'value', 'anomaly'])

results = pd.read_csv("traffic_data/real_world_data_results.csv")
performance_evaluation(ground_truth, results)