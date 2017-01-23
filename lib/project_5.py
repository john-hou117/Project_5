import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.feature_selection import SelectFromModel
import sklearn

def load_data_from_database():
    """ Function that loads data from a remote database.
    
    This function connects to a remote database server via Postgresql. It then reads the database
    into a Pandas DataFrame
    
    Inputs: None
    Outputs: Pandas DataFrame
    """
    engine = create_engine('postgresql://dsi:correct horse battery staple@joshuacook.me:5432')
    
    our_df = pd.read_sql_table('madelon', con=engine, index_col='index')
    
    return our_df


def add_to_process_list(process, data_dict):
    """ Function that adds a process (e.g. a transformer, or a model) to a process list.
    
    This function adds a "processes" key, which holds a "processes" list, to our data dictionary.
    The "processe"' list contains all the transformers and models that have been used on our data.
    This is similar to the "steps" property in sklearn's Pipeline class.
    
    Inputs: a process, a data dictionary
    Outputs: our updated data dictionary with the relevant process added to the dictionary
    """
    
    if 'processes' in data_dict.keys():
        data_dict['processes'].append(process)
    else:
        data_dict['processes'] = [process]
    
    return data_dict


def make_data_dict(our_df, random_state=None):
    """Function which makes a data dictionary that will hold all our important information.
    
    This function splits our received Pandas Dataframe into a target, y, and our feature matrix, X.
    Then, the target and feature matrix are split into training and test sets.
    
    Inputs: Pandas DataFrame
    Outputs: a data dictionary, containing our split X_train, X_test, y_train, and y_test
    """
    
    y = our_df['label']
    X = our_df.drop('label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    data_dict = {'X_train' : X_train,
           'X_test' : X_test,
           'y_train' : y_train,
           'y_test' : y_test}
    
    return data_dict


def validate_dictionary(data_dict_to_validate):
    """Function that validates, or in other words, ensures that our data dictionary has all the relevant
    parts.
    
    This function checks to see that our data dictionary has an X_train, X_test, y_train, and y_test. If
    it is missing any of these 4 components, a ValueError will be raised, informing the user as to which
    component they are missing.
    
    Inputs: data dictionary
    Outputs: a validated data dictionary, still containing our X_train, X_test, y_train, and y_test
    """
    
    try:
        data_dict_to_validate['X_train']
    except:
        raise ValueError('You need to pass an X_train')
        
    try:
        data_dict_to_validate['X_test']
    except:
        raise ValueError('You need to pass an X_test')
        
    try:
        data_dict_to_validate['y_train']
    except:
        raise ValueError('You need to pass a y_train')
        
    try:
        data_dict_to_validate['y_test']
    except:
        raise ValueError('You need to pass a y_test')
        

def general_transformer(transformer_of_choice, data_dict):
    """Function that transforms the data in the data dictionary.
    
    This function accepts any valid transformer (e.g. StandardScaler, SelectKBest, etc.), performs that
    transformation on the data, and then updates our data dictionary with the transformed data.
    
    Inputs: a transformer of our choosing, a data dictionary
    Outputs: data dictionary with transformed data (via X_train and X_test), and untransformed
    y_train, y_test
    """
    
    validate_dictionary(data_dict)
    
    transformed_data_dict = dict(data_dict)
    
    transformer_of_choice.fit(transformed_data_dict['X_train'], transformed_data_dict['y_train'])
    
    transformed_data_dict['X_train'] = transformer_of_choice.transform(transformed_data_dict['X_train'])
    transformed_data_dict['X_test'] = transformer_of_choice.transform(transformed_data_dict['X_test'])
    
    transformed_data_dict = add_to_process_list(transformer_of_choice, transformed_data_dict)
    
    return transformed_data_dict


def general_model(model_of_choice, data_dict):
    """Function that runs a model on the data in the data dictionary.
    
    This function accepts any valid model (e.g. LogisticRegression, KNeighborsClassifier, etc.),
    runs the data through the model, and then updates our data dictionary with train and test scores.
    
    Inputs: a model of our choosing, a data dictionary
    Outputs: data dictionary with our transformed X_train, transformed X_test, untransformed y_train and
    y_test, processes containing a list of our transformer and model, and our train and test scores
    """
    
    validate_dictionary(data_dict)
    
    model_data_dict = dict(data_dict)
    
    model_of_choice.fit(model_data_dict['X_train'], model_data_dict['y_train'])
    
    model_data_dict['train_score'] = model_of_choice.score(model_data_dict['X_train'], model_data_dict['y_train'])
    model_data_dict['test_score'] = model_of_choice.score(model_data_dict['X_test'], model_data_dict['y_test'])
    
    if type(model_of_choice) != sklearn.neighbors.classification.KNeighborsClassifier:
    # could not figure out how to make KNeighborsClassifier work with SelectFromModel. Must fix for
    # future use
    
        sfm = SelectFromModel(model_of_choice, prefit=True)
        # need to figure out if the following line is necessary: sfm.fit(model_data_dict['X_train'], model_data_dict['y_train'])
    
        sal_features = sfm.transform(model_data_dict['X_train'])
    
        model_data_dict['sal_features'] = sal_features
    
        model_data_dict['coef_'] = model_of_choice.coef_

    model_data_dict = add_to_process_list(model_of_choice, model_data_dict)
    
    return model_data_dict