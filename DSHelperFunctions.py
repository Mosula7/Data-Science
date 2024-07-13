import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, ConfusionMatrixDisplay, classification_report
from scipy.stats import ks_2samp
from operator import itemgetter


def eda_uni(df, col, dtype='cont'):
    if dtype == 'cont':
        sns.set_style('whitegrid')

        mean = df[col].mean()
        min = df[col].min()
        max = df[col].max()
        std = df[col].std()
        median = df[col].median()

        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = q3 - q1

        missing = df[col].isna().sum()
        outliers = df[(df[col] < q1 - iqr * 1.5) |
                    (df[col] > q3 + iqr * 1.5)].shape[0]

        n_unique = df[col].nunique()
        n_bins = 24 if n_unique > 24 else n_unique
        y_max = np.histogram(df[col].dropna(), bins=n_bins)[0].max()

        fig, ax = plt.subplots(1,2, figsize=(20,7))

        sns.histplot(x=df[col],  bins=n_bins, kde=True, common_bins=False, common_norm=False, color='darkorange', ax=ax[0], edgecolor='black')
        ax[0].vlines(mean, ymin=0, ymax=y_max, label='mean', color='green', linewidth=3)
        ax[0].vlines(median, ymin=0, ymax=y_max, label='median', color='red', linewidth=3)
        ax[0].vlines(q1, ymin=0, ymax=y_max, label='Q1', color='grey', linewidth=3)
        ax[0].vlines(q3, ymin=0, ymax=y_max, label='Q3', color = 'black', linewidth=3)
        ax[0].set_title(f'mean = {mean:.2f},   std = {std:.2f},   min = {min:.2f},   max = {max:.2f},   missing = {missing}')
        ax[0].legend()

        sns.boxplot(data=df[col], ax=ax[1], color='darkorange')
        ax[1].set_title(f'median = {median:.2f},   Q1 = {q1:.2f},   Q3 = {q3:.2f},   outliers = {outliers}')

        fig.suptitle(col.title(), fontsize=20)
        plt.show()


    if dtype == 'cat':
        sns.set_style('whitegrid')

        colors = sns.color_palette('pastel')
        fig, ax = plt.subplots(1, 2, figsize=(20,7))
        sns.histplot(df[col].fillna('NA'), ax = ax[0], color='orange')
        ax[0].tick_params(axis='x', labelrotation = 90)

        col_val_count = df[col].value_counts(dropna=False)
        ax[1].pie(col_val_count, labels=col_val_count.index, colors=colors,  autopct='%1.1f%%')
        fig.suptitle(col.title(), fontsize=20)

        plt.tight_layout()
        plt.show()


def xy_plot(df:pd.DataFrame, *, x:str, y:str, x_type='cont', y_type = 'cat', figsize=(8,7.5)):

  if x_type == 'cont' and y_type == 'cat':

    classes = df[y].unique()

    fig, ax = plt.subplots(2,1, figsize=figsize)

    sns.boxplot(data=df, x=y, y=x, ax=ax[0], palette=sns.color_palette('pastel'))


    n_bins = len(np.unique(df[x])) if len(np.unique(df[x])) < 25 else 25
    sns.histplot(data=df, x=x, hue=y, bins=n_bins, stat="density", edgecolor='black', alpha=.3, ax=ax[1])

    fig.suptitle(x, fontsize=16)
    plt.tight_layout()

    stats_df = pd.DataFrame(index=classes, columns=['mean', 'std', 'median', 'Q1', 'Q3', 'IQR', 'outliers', 'missing'])
    sep_y = {class_: df.query(f'{y} == "{class_}"')[x] for class_ in classes}

    for class_, data in sep_y.items():
      mean = data.mean()

      q1 = data.quantile(.25)
      q3 = data.quantile(.75)
      iqr = q3 - q1

      stats_df.loc[class_, 'mean'] = mean
      stats_df.loc[class_, 'std'] = data.std()
      stats_df.loc[class_, 'median'] = data.median()
      stats_df.loc[class_, 'Q1'] = q1
      stats_df.loc[class_, 'Q3'] = q3
      stats_df.loc[class_, 'IQR'] = iqr
      stats_df.loc[class_, 'outliers'] = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
      stats_df.loc[class_, 'missing'] = data.isna().sum()

    plt.show()
    display(stats_df)
    print('-'*105)

  elif x_type == 'cat' and y_type == 'cat':
    if len(df[col].unique()) > 30:
      print(f'too many classes in "{x}" try other plots')
    else:

      fig, ax = plt.subplots(figsize=figsize)
      sns.countplot(data=df.fillna('NA'), x=x, hue=y, palette=sns.color_palette('pastel'), edgecolor='black', ax=ax)
      ax.set_title(x, fontsize=16)
      plt.show()


def corr_plots(df, figsize=(12,30)):
    fig, ax = plt.subplots(3, 1, figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues', fmt='.2f', mask=np.triu(df.corr(numeric_only=True)), ax=ax[0])
    ax[0].set_title('Correlation Heatmap', fontsize=16)

    sns.heatmap(df.isna().corr(), annot=True, cmap='Blues', fmt='.2f', mask=np.triu(df.isna().corr()), ax=ax[1])
    ax[1].set_title('Missing Value Correlation Heatmap', fontsize=16)


    sns.heatmap(df.isna().T, cmap='Blues', ax=ax[2])
    ax[2].axes.xaxis.set_ticklabels([])
    plt.show()


def split_data(df: pd.DataFrame, target: str, test_size: float, 
               val_size: float=None, random_state:int = 0):
    """
    returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not val_size:
        val_size = test_size / (1 - test_size)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val[target], random_state=random_state)

    X_train = train[train.columns.drop(target)]
    X_val = val[val.columns.drop(target)]
    X_test = test[test.columns.drop(target)]

    y_train = train[target]
    y_val = val[target]
    y_test = test[target]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def classification_predictive_power(y, pred, set_name=None, scoring_func=lambda x: 487.123 + 28.8539 * np.log((1 - x) / x), threshold=0.5):
    """
    makes 4 plots:
    first two plots are PDF and CDF of probabilities or scores (if additional scoring/transformation function is provided)
    the third plot is a confusion matrix and the final plot is a classification report
    the function also calculates KS statistic, AUC and accuracy
    """

    if not set_name:
        try:
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            set_name = [var_name for var_name, var_val in callers_local_vars if var_val is pred][0]
            set_name = set_name.replace('pred_', '').title() + ' Set'
        except:
            set_name = 'Set'

    title_font_size = 14
    table_cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=1,as_cmap=True)
    fig, ax = plt.subplots(2,2, figsize=(14,12))

    auc = roc_auc_score(y, pred)


    # KS
    ks_data = pd.DataFrame({'Target': y, 'prob': pred})
    ks_data['SCORE'] = ks_data['prob'].apply(scoring_func)
    sns.histplot(data=ks_data, x='SCORE', hue='Target',  stat='probability', kde=True, bins=20, common_bins=False, 
                 common_norm=False, palette=['darkorange', 'grey'], ax=ax[0][0], edgecolor='black')
    ax[0][0].set_title('PDF', fontsize=title_font_size)

    fpr, tpr, _ = roc_curve(y, pred)
    
    ax[0][1].plot(fpr, tpr, 'orange', label = 'AUC = %0.2f' % auc)
    ax[0][1].set_title('ROC')
    
    ax[0][1].legend(loc = 'lower right')
    ax[0][1].plot([0, 1], [0, 1], 'grey', linestyle='--')
    ax[0][1].set_xlim([0, 1])
    ax[0][1].set_ylim([0, 1])
    ax[0][1].set_ylabel('True Positive Rate')
    ax[0][1].set_xlabel('False Positive Rate')

    #confusion matrix
    ax[1][0].grid(None)
    ConfusionMatrixDisplay.from_predictions(y, pred > threshold, ax=ax[1][0], cmap=table_cmap)
    ax[1][0].grid(None)
    ax[1][0].set_title('Confusion Matrix', fontsize=title_font_size)
  
    # classification report
    cr = pd.DataFrame(itemgetter(*tuple(y.unique().astype('str')))(classification_report(y, pred > threshold, output_dict=True))).drop(columns = 'support')
    sns.heatmap(cr,annot=True,vmax=1,fmt='.5f',ax=ax[1][1],
                cmap=table_cmap)
    ax[1][1].set_title('Classification Report', fontsize=title_font_size)

    # KS, AUC, Accuracy
    fig.suptitle(f""" {set_name} \n   
                 KS - {ks_2samp(ks_data.query('Target == 0')['SCORE'], ks_data.query('Target == 1')['SCORE'])[0]:.3f}\
                 GINI - {2 * auc - 1:.3f}\
                 Accuracy - {accuracy_score(y, pred>.5):.3f}""", fontsize=16) # y=.95, 

    # formating 
    for _, spine in ax[1][1].spines.items():
        spine.set_visible(True)

    cbar = ax[1][1].collections[0].colorbar
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.75)

    ax[0][0].set_axisbelow(True)
    ax[0][0].yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)
    ax[0][1].set_axisbelow(True)
    ax[0][1].yaxis.grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.5)

    ax[0][0].spines[['left', 'right', 'top']].set_visible(False)
    ax[0][1].spines[['left', 'right', 'top']].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def objective_lgbm(trial, param_grid, X, y, metrics):   
    # hyperparams for the model
    params = {}
    for key, value in param_grid.items():
        if isinstance(value, list) or isinstance(tuple, list):
            params[key] = trial.suggest_categorical(key, value)
        else:
            params[key] = value
    # arrays for metrics
    auc_array = np.array([])
    ks_array = np.array([])

    # doing 5 fold stratified cross validation
    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(skf.split(X=X, y=y)):

        # splitting data
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        X_valid = X.iloc[valid_index]
        y_valid = y.iloc[valid_index]
        
        # initializing, fitting and predicting on validation set
        lgbm = lgb.LGBMClassifier(**params)
        lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        y_pred = lgbm.predict_proba(X_valid)[:,-1]

        # calculating auc
        auc = roc_auc_score(y_valid, y_pred)
        auc_array = np.append(auc_array, auc)
        
        # calculating ks
        ks_data = pd.DataFrame({'Target': y_valid, 'Prediction': y_pred})
       
        ks = ks_2samp(ks_data.query('Target == 0')['Target'], ks_data.query('Target == 1')['Target'])[0]
        ks_array = np.append(ks_array, ks)

        # logging results into the dataframe
        metrics.loc[str(params),f'auc_{i}'] = auc
        metrics.loc[str(params),f'ks_{i}'] = ks
      
    metrics.loc[str(params), f'avg_auc'] = np.mean(auc_array)
    metrics.loc[str(params), f'avg_ks'] = np.mean(ks_array)
    
    #returning main point of interest, auc, for the objective function
    return np.mean(auc_array)
