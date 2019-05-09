import pandas as pd


def coef_df(pipeline):
    vect = pipeline.named_steps['vect']
    clf = pipeline.named_steps['clf']
    features = vect.get_feature_names()
    coef = clf.coef_
    coef_df = pd.DataFrame({'name': features, 'coef': coef.ravel()})
    coef_df.sort_values('coef', inplace=True)
    return coef_df
