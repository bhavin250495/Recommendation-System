import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
import implicit

raw_data = pd.read_csv('usersha1-artmbid-artname-plays.tsv',sep='\t')
raw_data = raw_data.drop(raw_data.columns[1],axis=1)
raw_data.columns = ['users','artists','plays']

data = raw_data.dropna().head(20)

data['user_id'] = data['users'].astype('category').cat.codes
data['artist_id'] = data['artists'].astype('category').cat.codes


item_lookup = data[['artists','artist_id']].drop_duplicates()
item_lookup['artist_id'] = item_lookup['artist_id'].astype('str')

data = data.drop(['artists','users'],axis=1)

data = data.loc[data.plays != 0]

users = list(np.sort(data.user_id.unique()))
artists = list(np.sort(data.artist_id.unique()))
plays = list(data.plays)

rows = data.user_id.astype(int)
cols = data.artist_id.astype(int)

data_sparse = sparse.csr_matrix((plays,(rows,cols)),shape=(len(users),len(artists)))

user_vecs, item_vecs = implicit_als(data_sparse, iterations=20, features=20, alpha_val=40)


model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=50)
alpha = 15
data = (data_sparse * alpha).astype('double')
model.fit(data)
item_vecs1 = model.item_factors
user_vecs1 = model.user_factors
item_vecs = item_vecs1


def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):

    """ Implementation of Alternating Least Squares with implicit data. We iteratively
        compute the user (x_u) and item (y_i) vectors using the following formulas:

        x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
        y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

        Args:
            sparse_data (csr_matrix): Our sparse user-by-item matrix

            alpha_val (int): The rate in which we'll increase our confidence
            in a preference with more interactions.

            iterations (int): How many times we alternate between fixing and
            updating our user and item vectors

            lambda_val (float): Regularization value

            features (int): How many latent features we want to compute.

        Returns:
            X (csr_matrix): user vectors of size users-by-features

            Y (csr_matrix): item vectors of size items-by-features
         """

    # Calculate the foncidence for each value in our data
    confidence = sparse_data * alpha_val

    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape

    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size=(user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size=(item_size, features)))

    # Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)

    I = sparse.eye(features)
    lI = lambda_val * I
    # Start main loop. For each iteration we first compute X and then Y
    for i in range(iterations):
        print('iteration %d of %d' % (i + 1, iterations))

        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in range(user_size):
            # Get the user row.
            u_row = confidence[u, :].toarray()

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

        for i in range(item_size):
            # Get the item column and transpose it.
            i_row = confidence[:, i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y



item_id = 229

item_vec = item_vecs[item_id].T

scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]

scores = item_vecs.dot(item_vecs)
scores = scores.reshape(1,-1)[0]

top_10 = np.argsort(scores)[::-1][:10]


artists = []
artists_score = []

for idx in top_10:
    print(idx)
    print(item_lookup.artists[item_lookup.artist_id == idx])
    artist = item_lookup.artists[item_lookup.artist_id == idx].iloc[0]
    print(item_lookup.artists[item_lookup.artist_id == idx])

    artists.append(artist)
    artists_score.append(scores[idx])

similar = pd.DataFrame({'Artist':artists,'Score':artists_score})

