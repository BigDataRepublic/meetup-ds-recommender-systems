import numpy as np

plot_histograms = ", ".join(["df.groupby('item').count()['rating'].hist(bins=100, ax=ax0)",
                             "df.groupby('user').count()['rating'].hist(bins=100, ax=ax1)",
                             "ax0.set_title('Users per item')",
                             "ax1.set_title('Items per user')"])

get_n_ratings = 'len(df)'
get_n_users = "df['user'].nunique()"
get_n_items = "df['item'].nunique()"

n_ratings = 76773
n_users = 943
n_items = 1566

compute_sparsity = 'n_ratings / (n_users * n_items)'

user0_top3 = np.array([ 49, 180, 167])

predict_user0 = 'W[0,:].dot(H)'

reconstruction_error = 'np.sqrt(np.sum((W.dot(H) - X) ** 2))'

area_under_the_curve = "auc(series.index, series.values) / 100"
auc_score = 0.86

mrr = "np.apply_along_axis(lambda r: 1 / r, axis=0, arr=x).mean()"
mrr_score = 0.03

item_similarities = 'cosine_similarity(model.H.T)'


# # pointers for implementing weighted ALS
#
# for k in range(C.shape[0]):
#     W[k, :] = X[k, :].dot(np.diag(C[k, :])).dot(H.T).dot(np.linalg.pinv(H.dot(np.diag(C[k, :])).dot(H.T) + l2 * np.eye(n_components)))
# for k in range(C.shape[1]):
#     H[:, k] = np.linalg.pinv(W.T.dot(np.diag(C[:, k])).dot(W) + l2 * np.eye(n_components)).dot(W.T).dot(np.diag(C[:, k])).dot(X[:, k])

