class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def fix_types(df_train, df_valid, df_test):
    df_train = df_train.astype({col: float for col in df_train.columns if col not in ["time", "true_time", "event"]})
    df_train["time"] = df_train["time"].astype(int)
    df_train["true_time"] = df_train["true_time"].astype(int)
    df_train["event"] = df_train["event"].astype(bool)
    df_valid = df_valid.astype({col: float for col in df_valid.columns if col not in ["time", "true_time", "event"]})
    df_valid["time"] = df_valid["time"].astype(int)
    df_valid["true_time"] = df_valid["true_time"].astype(int)
    df_valid["event"] = df_valid["event"].astype(bool)
    df_test = df_test.astype({col: float for col in df_train.columns if col not in ["time", "true_time", "event"]})
    df_test["time"] = df_test["time"].astype(int)
    df_test["true_time"] = df_test["true_time"].astype(int)
    df_test["event"] = df_test["event"].astype(bool)
    return df_train, df_valid, df_test