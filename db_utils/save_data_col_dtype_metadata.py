import os
import pandas as pd
import pickle

def main():
    """
    utility function:
    saves out the datatype of each column in the training_data
    into a file that's used in the main program

    only has to be rerun in the future when a new column is added to the training_data

    """
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

        except:
            continue

    print(col_dtypes)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TDDIR = os.path.join(BASE_DIR,'iowaHomes/predict/predictModel/predictionModels/training_data/col_dtypes.sav')
    pickle.dump(col_dtypes, open(TDDIR, 'wb'))
main()

