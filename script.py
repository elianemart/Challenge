import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
path = 'data/'

product = str(input('Enter a product (P1 to P8): ')) 
value = float(input('Enter a value for this product: ')) 

z_table = pd.read_csv(path+"z_table.csv").rename(columns={'UNIT_VALUE': 'mean', 'UNIT_VALUE.1': 'std'})

# load the model
filename = 'trained_model_{}.sav'.format(product)
model = pickle.load(open(path+filename, 'rb'))

z_mean = z_table.loc[z_table['PROD_ID'] == product]['MEAN'].values[0]
z_std = z_table.loc[z_table['PROD_ID'] == product]['STD'].values[0]

x_test = (value-z_mean)/z_std
x_test = x_test.reshape(-1, 1)

polyformer  = PolynomialFeatures(degree=2, include_bias=True)
x_test_model = polyformer.fit_transform(x_test)
y_pred = model.predict(x_test_model)[0]

if y_pred < 0:
    print("Value very different from normal price, unknown behavior, most likely price is: ", z_mean)
else: 
    print('The total revenue predict was: ', round(y_pred))
