import os
import pandas as pd
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVPATH = os.path.join(BASE_DIR,'db_utils/iowaHomes.csv')
data = pd.read_csv(CSVPATH)


col_dtypes = {}
import re
for key in list(data.columns):
    # converting/matching user input data to training data dtypes
    key_dtype = re.sub(r'\d+', '',  type(data[key][0]).__name__)
    try:
        if key_dtype == "int":
            col_dtypes[key] = "int"
        elif key_dtype == "float":
            col_dtypes[key] = "float"
        else:
            col_dtypes[key] = "float"
    except:
        continue
#
# df = pd.DataFrame(columns=['col','dtype'])
# for idx,col in enumerate(col_dtypes):
#     df.loc[idx] = [col, col_dtypes[col]]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDDIR = os.path.join(BASE_DIR,'iowaHomes/predict/predictModel/predictionModels/training_data/col_dtypes.sav')
pickle.dump(col_dtypes, open(TDDIR, 'wb'))


import sqlite3
#conn = sqlite3.connect(DB_DIR)
#print(df)
#conn.execute()
#df.to_csv("",index=False)
