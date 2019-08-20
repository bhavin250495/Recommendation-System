import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sparse

# Read all data
aisles = pd.read_csv('instacart-market/aisles.csv')
departments = pd.read_csv('instacart-market/departments.csv')
order_products_prior = pd.read_csv('instacart-market/order_products__prior.csv')
order_products_train = pd.read_csv('instacart-market/order_products__train.csv')
orders = pd.read_csv('instacart-market/orders.csv')
products = pd.read_csv('instacart-market/products.csv')

# Only select prior data
orders_prior = orders[orders.eval_set == 'prior']

orders_prior_test = order_products_prior.drop(['reordered'],axis = 1)

order_test = orders_prior[['order_id','user_id']]

lookup_table = pd.merge(order_test,orders_prior_test,on='order_id',how='inner')
lookup_table_grouped = lookup_table.drop(['order_id'],axis=1)


lookup_table_grouped = lookup_table_grouped.groupby(['user_id','product_id']).sum().reset_index()

lookup_table_grouped['user_idx'] =  lookup_table_grouped['user_id'].astype('category').cat.codes
lookup_table_grouped['item_idx'] =  lookup_table_grouped['product_id'].astype('category').cat.codes

user_ids = lookup_table_grouped['user_idx'].drop_duplicates().sort_values()
item_ids = lookup_table_grouped['item_idx'].drop_duplicates().sort_values()

rows = lookup_table_grouped.user_idx.astype(int)

add_to_cart = lookup_table_grouped['add_to_cart_order'].astype(float)

user_to_item = sparse.csr_matrix( (add_to_cart,(lookup_table_grouped['user_idx'],lookup_table_grouped['item_idx'])) )

item_to_user =  sparse.csr_matrix( (add_to_cart,(lookup_table_grouped['item_idx'],lookup_table_grouped['user_idx'])) )

item_lookup = lookup_table_grouped[['product_id','item_idx']].drop_duplicates().sort_values(['item_idx'])
item_lookup = pd.merge(item_lookup,products,on='product_id',how='inner')

model = implicit.als.AlternatingLeastSquares(factors=20,regularization=0.1,iterations=50)
alpha = 15
data_conf = (item_to_user * alpha).astype('double')
model.fit(data_conf)

### Similar item
similar_items = model.similar_items(0,10)

for item in similar_items:
    idx,score = item
    name = item_lookup.product_name[item_lookup.item_idx == idx]

    print('Score:: %.3f '% (score) + name.values.item() )

### Recommendation
user_id = 0
recommendations = model.recommend(user_id,user_to_item)

for item in recommendations:
    idx,score = item
    name = item_lookup.product_name[item_lookup.item_idx == idx]

    print('Score:: %.3f '% (score) + name.values.item())


