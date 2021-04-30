import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance

def plotROC(y_test, y_pred_score, filename):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_score, drop_intermediate=False, pos_label=1)
    plt.figure(figsize=(3,3))
    plt.rcParams['font.size'] = 8
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    s = 'AUC = %0.2f' % roc_auc_score(y_test, y_pred_score)
    plt.text(0.6, 0.15, s, fontsize=8)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(filename, dpi=600)

def plotPRcurve(estimator, X, y, y_pred_score, filename):
    fig, ax = plt.subplots(figsize=(3,3))
    average_precision = average_precision_score(y, y_pred_score)
    #print('Average precision-recall score: {0:0.2f}'.format(
    #      average_precision))
    disp = plot_precision_recall_curve(estimator, X, y, ax=ax)
    plt.rcParams['font.size'] = 8
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    s = 'AP = %0.2f' % average_precision
    plt.text(0.6, 0.15, s, fontsize=8)
    #disp.ax_.set_title('2-class Precision-Recall curve: '
    #                   'AP={0:0.2f}'.format(average_precision))
    baseline = len(y[y==1]) / len(y)
    plt.plot([0, 1], [baseline, baseline], linestyle='--') #, label='Baseline'
    plt.legend('',frameon=False)
    fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(filename, dpi=600)

def getPermutationFeatureImportance(estimator, X, y, filename, N_repeats=30):
    result = permutation_importance(estimator, X, y, n_repeats=N_repeats, random_state=0, n_jobs=-1)
    fig, ax = plt.subplots(figsize=(5,3))
    plt.rcParams['font.size'] = 8
    ax.boxplot(result.importances.T,
               vert=False, labels=['Gene expression level', 'Gene expression variance', 'Co-expression degree', 'Codon adaptation index',
       'Sequence variation', 'Synthetic sick/lethal paralogs in S. cer',
       'Ortholog essentiality in S. cer', 'Length', 'Hits', 'Reads',
       'Freedom index', 'Neighborhood index', 'Upstream hits 100'])
    plt.xlabel('Importance')
    fig.tight_layout()
    plt.savefig(filename, dpi=600)

def plotGRACEv2Distributionplot(df_6638_beforeImpute, df_GRACEv2):
    plt.figure(figsize=(3,3))
    plt.hist(df_6638_beforeImpute.loc[df_GRACEv2.index]['RF_prediction_proba'],bins=30,color='orange')
    plt.xlabel('Prediction scores')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rf_dist_GRACEv2_30bins.pdf', dpi=600)

def plotRFScoreDistribution(df_6638_beforeImpute):
    plt.figure(figsize=(3,3))
    plt.hist(df_6638_beforeImpute['RF_prediction_proba'],bins=30)
    plt.xlabel('Prediction scores')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rf_dist_30bins.pdf', dpi=600)
    
def applyRandomForestClassifier_GridSearchCV(X, y, grid_parameters, num_cv=5, testSize=0.2, randomState=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    train_test_list = [X_train, X_test, y_train, y_test]
    rf_clf = RandomForestClassifier()
    grid_search_clf = GridSearchCV(rf_clf, grid_parameters, cv = num_cv, n_jobs=-1, verbose=True)
    grid_search_clf.fit(X_train, y_train)
    best_grid_clf = grid_search_clf.best_estimator_
    
    y_pred = best_grid_clf.predict(X_test)
    y_pred_score = best_grid_clf.predict_proba(X_test)[:,1]
    print("The optimal model performance during the GridSearchCV is")
    #print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    #print(accuracy_score(y_test, y_pred))

    plotROC(y_test, y_pred_score, 'rf_GRACEtest_ROC.pdf')
    plotPRcurve(best_grid_clf, X_test, y_test, y_pred_score, 'rf_GRACEtest_PR.pdf')
    
    return best_grid_clf, train_test_list

def plotGRACEv2Density(df_6638_beforeImpute):
    plt.figure(figsize=(3,3))
    plt.rcParams['font.size'] = 8
    # Remove the GRACEv2 screens that were once tested by GRACE
    df_GRACEv2_score = df_6638_beforeImpute[df_6638_beforeImpute['GRACEv2'].notnull()]
    df_GRACEv2_score = df_GRACEv2_score[df_GRACEv2_score['GRACE'].isnull()]
    df_GRACEv2_ESS = df_GRACEv2_score[df_GRACEv2_score['GRACEv2']==1]
    df_GRACEv2_NE = df_GRACEv2_score[df_GRACEv2_score['GRACEv2']==-1]
    p1 = sns.kdeplot(df_GRACEv2_NE['RF_prediction_proba'], shade=True, color="b")
    p2 = sns.kdeplot(df_GRACEv2_ESS['RF_prediction_proba'], shade=True, color="r")
    plt.legend(['Non-essential','Confirmed essential'])
    plt.xlabel('Prediction scores for validation candidates')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('rf_GRACEv2_density.pdf', dpi=600)

def plotGRACE_ROC_PR(train_test_list):
    #X_train=train_test_list[0]
    X_test=train_test_list[1]
    #y_train=train_test_list[2]
    y_test=train_test_list[3]
    # Please make sure that the best rf model has already been built and dumped in the same directory
    clf = load('rf_clf_best.joblib')
    y_pred = clf.predict(X_test)
    y_pred_score = clf.predict_proba(X_test)[:,1]
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    plotROC(y_test, y_pred_score, 'rf_GRACEtest_ROC.pdf')
    plotPRcurve(clf, X_test, y_test, y_pred_score, 'rf_GRACEtest_PR.pdf')

if __name__ == "__main__":
    # Read in the 6,638 genes with 13 features and essentiality labels from GRACE & GRACEv2
    # By default, the input file is in the same directory as this program
    direc = './Calbicans_13Features_6638genes_beforeImputation_210302.tsv'
    df_6638_beforeImpute = pd.read_csv(direc, sep='\t', index_col=0)

    # Impute the missing values with the mean of the corresponding attributes
    df_6638_features_imputed = df_6638_beforeImpute.iloc[:,:13].fillna(df_6638_beforeImpute.iloc[:,:13].mean())
    df_6638_features_imputed[['GRACE', 'GRACEv2']] = df_6638_beforeImpute[['GRACE', 'GRACEv2']]

    # DataFrame for GRACE genes
    df_GRACE = df_6638_features_imputed[df_6638_features_imputed['GRACE'].notnull()]
    
    # DataFrame for GRACEv2 genes - remove the 7 rebuilt GRACE strains
    df_GRACEv2 = df_6638_features_imputed[df_6638_features_imputed['GRACEv2'].notnull()]
    df_GRACEv2 = df_GRACEv2[df_GRACEv2['GRACE'].isnull()]
    
    # Train and optimize on the GRACE genes, and then validate on the GRACEv2 genes
    grid_parameters = {
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True],
        'oob_score': [True],
        'max_depth': [i for i in range(1,11)],
        'max_samples': [0.1, 0.25, 0.3333, 0.5, 0.6667, 0.75, 0.9],
        'max_features': ['sqrt',  0.5, 0.75],
        'n_estimators': [100, 150, 200, 250, 300]
    }
    
    X_GRACE = df_GRACE.iloc[:,:13].values
    y_GRACE = df_GRACE['GRACE'].values
    best_grid_clf, train_test_list = applyRandomForestClassifier_GridSearchCV(X_GRACE, y_GRACE, grid_parameters)
    dump(best_grid_clf, 'rf_clf_best.joblib')

    # Use the optimal rf model to fit the whole GRACE set (2,327 genes)
    best_grid_clf.fit(X_GRACE, y_GRACE)
    # Get the permutation feature importance for this estimator on the training (GRACE) set
    # Permutation for each feature will be repeated 30 times
    getPermutationFeatureImportance(best_grid_clf, X_GRACE, y_GRACE, 'perm_importance_GRACE.pdf')
    
    X_GRACEv2 = df_GRACEv2.iloc[:,:13].values
    y_GRACEv2 = df_GRACEv2['GRACEv2'].values
    
    y_GRACEv2_pred = best_grid_clf.predict(X_GRACEv2)
    y_GRACEv2_pred_score = best_grid_clf.predict_proba(X_GRACEv2)[:,1]
    print("The validation performance on the experimental GRACEv2 dataset is")
    #print(confusion_matrix(y_GRACEv2, y_GRACEv2_pred))
    print(classification_report(y_GRACEv2, y_GRACEv2_pred))
    #print(accuracy_score(y_GRACEv2, y_GRACEv2_pred))

    plotROC(y_GRACEv2, y_GRACEv2_pred_score, 'rf_GRACEv2_ROC.pdf')
    plotPRcurve(best_grid_clf, X_GRACEv2, y_GRACEv2, y_GRACEv2_pred_score, 'rf_GRACEv2_PR.pdf')

    # Predict essentiality for 6,638 genes
    y_6638_pred = best_grid_clf.predict(df_6638_features_imputed.iloc[:,:13].values)
    y_6638_pred_score = best_grid_clf.predict_proba(df_6638_features_imputed.iloc[:,:13].values)[:,1]
    
    df_6638_beforeImpute['RF_prediction'] = y_6638_pred
    df_6638_beforeImpute['RF_prediction_proba'] = y_6638_pred_score
    df_6638_beforeImpute.to_csv('./Calbicans_13Features_6638genes_RF_predictions.tsv', sep='\t')

