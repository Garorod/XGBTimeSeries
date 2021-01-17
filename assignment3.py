# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# %%
x_cols = ['Loan_age', 'CreditScore',
       'FullDocumentation', 'State_CA', 'State_FL', 'State_NY', 'State_NJ',
       'State_GA', 'Spread', 'Spread_squared', 'Remaining_balance', 'LTV',
       'spring_summer']
indicator_cols = ['Default_indicator', 'Prepayment_indicator']
y_cols = ['status']


# %%
loans_df = pd.read_csv('notebooks/FRM_perf.csv')
is_active = loans_df[['Default_indicator', 'Prepayment_indicator']].sum(axis=1) == 0
loans_df['status'] = loans_df[indicator_cols].assign(Active=is_active.astype(int)).idxmax(axis=1).str.replace('_indicator', '').astype('category')
categories = loans_df['status'].cat.categories

train_loan_id, test_loan_id = train_test_split(loans_df['Loan_id'].unique(), test_size=0.1, random_state=0)
train_loans = loans_df.loc[loans_df['Loan_id'].isin(train_loan_id)]
test_loans = loans_df.loc[loans_df['Loan_id'].isin(test_loan_id)]


# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import ADASYN, RandomOverSampler

clf = RandomForestClassifier(max_depth=5)
X_resampled, y_resampled = RandomOverSampler().fit_resample(train_loans[x_cols], train_loans[y_cols])
clf.fit(X_resampled, y_resampled.ravel())


# %%
calc_auc = lambda df: roc_auc_score(df[y_cols].squeeze(), clf.predict_proba(df[x_cols]), multi_class='ovr')
calc_auc(train_loans), calc_auc(test_loans)


# %%
calc_avg_prob = lambda df: pd.DataFrame(clf.predict_proba(df[x_cols]), columns=categories).assign(status=df[y_cols].values).pivot_table(index='status')
calc_avg_prob(train_loans), calc_avg_prob(test_loans)


# %%
pd.DataFrame(clf.predict_proba(test_loans[x_cols]), columns=categories).plot()


# %%
y_resampled['status'].value_counts()


# %%
train_loans[y_cols]['status'].value_counts()


# %%
import importlib
neuraltree = importlib.import_module('Neural-Tree.Layers.model')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()


# %%
tree = neuraltree.SoftDecisionTree(max_depth=6,n_features=len(x_cols),n_classes=len(categories),max_leafs=None)
tree.build_tree()

optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(tree.loss)


# %%
def make_loans_batches(loans, batch_size):
    num_batches = int(len(loans) / batch_size + 0.5)
    scrambled = loans.sample(frac=1.0)
    def _gen():
        for ith_batch in range(num_batches):
            yield scrambled.iloc[ith_batch * batch_size:(ith_batch+1) * batch_size]
    return num_batches, _gen()


# %%
init = tf.global_variables_initializer()

EPOCHS = 10000
display_step = 100
batch_size = 32
val_batch_size = 32
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        avg_cost = 0.
        # Loop over all batches
        acc =0.0
        val_acc = 0.0
        num_batch, loan_batch_gen = make_loans_batches(train_loans, batch_size)
        val_num_batch, val_loan_batch_gen = make_loans_batches(test_loans, val_batch_size)
        for i in range(num_batch):
            train_loan_batch = next(loan_batch_gen)
            batch_xs, batch_ys = train_loan_batch[x_cols], pd.get_dummies(train_loan_batch[y_cols]['status'], columns=[0, 1, 2])

            c = tree.boost(X=batch_xs,y=batch_ys,sess=sess, optimizer=optimizer, tree=tree)


            target = batch_ys.idxmax(axis=1).map(dict(zip(categories, range(3))))
            preds = tree.predict(X=batch_xs,y=batch_ys,sess=sess)
            acc += accuracy_score(y_pred=preds,y_true=target)/num_batch

            # Compute average loss

            avg_cost+= acc/num_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            batch_val_xs, batch_val_ys = mnist.validation.next_batch(val_batch_size)


            val_target = np.argmax(batch_val_ys, axis=1)
            val_preds = tree.predict(X=batch_val_xs,y=batch_val_ys,sess=sess)
            val_acc = accuracy_score(y_pred=val_preds, y_true=val_target)
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                    "{:.9f}".format(avg_cost),"training_accuracy=","{:.4f}".format(acc),
                    "validation_accuracy=","{:.4f}".format(val_acc)  )
            #print(collections.Counter(np.argmax(path_probs,axis=1)))

            #print(confusion_matrix(y_true=val_target,y_pred=val_preds) )


