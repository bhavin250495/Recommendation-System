import numpy as np
import pandas as pd
import scipy.sparse as sparse
import implicit
from sklearn.preprocessing import MinMaxScaler



articles = pd.read_csv('shared_articles.csv')
articles = articles[articles['eventType']=='CONTENT SHARED']
articles.drop(['authorUserAgent','authorRegion','authorCountry'],axis=1,inplace=True)
articles.drop(['eventType'],axis=1,inplace=True)

interactions = pd.read_csv('users_interactions.csv')
interactions.drop(['userAgent','userRegion','userCountry'],axis=1,inplace=True)

df = pd.merge(interactions[['eventType','contentId','personId']],articles[['contentId','title']],on='contentId',how='inner')


event_type_strength = {
    'VIEW' : 1.0,
    'LIKE' : 2.0,
    'BOOKMARK' : 2.5,
    'FOLLOW' : 3.0,
    'COMMENT CREATED' : 4.0
}

df['eventStrength'] = df['eventType'].apply(lambda x : event_type_strength[x])

# Grouping and summing of multiple events strength on same book
grouped_df = df.groupby(['contentId','personId','title']).sum().reset_index()

# encode ids into ints
grouped_df['content_id'] = grouped_df.contentId.astype('category').cat.codes
grouped_df['person_id'] = grouped_df.personId.astype('category').cat.codes
grouped_df.title = grouped_df.title.astype('category')

sparse_content_person = sparse.csr_matrix((grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
spare_person_content = sparse.csc_matrix((grouped_df['eventStrength'].astype(float),(grouped_df['person_id'], grouped_df['content_id'])))

model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=50)
alpha = 15
data = (sparse_content_person * alpha).astype('double')
model.fit(data)

content_id = 61
n_similar = 10

person_vecs = model.user_factors
content_vecs = model.item_factors


content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))

scores = content_vecs.dot(content_vecs[content_id]) / content_norms

top_idx1 = np.argpartition(scores, -n_similar)[-n_similar:]

top_idx = np.argsort(scores)[-n_similar:]

similar = sorted(zip(top_idx, scores[top_idx] / content_norms[content_id]), key=lambda x: -x[1])

for content in similar:
    idx,score = content
    print(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])

def recommend(person_id,sparse_person_content,person_vecs,content_vecs,num_contents=10):

    person_interactions = sparse_person_content[person_id,:].toarray()

    person_interactions = person_interactions.reshape(-1) + 1

    person_interactions[person_interactions > 1] = 0


    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray()

    min_max = MinMaxScaler()

    rec_vector_scaled =  min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]

    recommend_vector = person_interactions * rec_vector_scaled

    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]

    titles = []
    scores= []

    for id in content_idx:
        titles.append(grouped_df.title.loc[grouped_df.content_id == id].iloc[0])
        scores.append(recommend_vector[id])

    recommendations = pd.DataFrame({'Title':titles,'Scores':scores})

    return recommendations


person_vecs = sparse.csr_matrix(model.user_factors)
content_vecs = sparse.csr_matrix(model.item_factors)

person_id = 50


recommendations = recommend(person_id,spare_person_content,person_vecs,content_vecs)

print(recommendations)

