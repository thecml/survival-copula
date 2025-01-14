class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def format_data_deephit_single(train_dict, valid_dict, labtrans):
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].cpu().numpy()
    train_dict_dh['E'] = train_dict['E'].cpu().numpy()
    train_dict_dh['T'] = train_dict['T'].cpu().numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].cpu().numpy()
    valid_dict_dh['E'] = valid_dict['E'].cpu().numpy()
    valid_dict_dh['T'] = valid_dict['T'].cpu().numpy()
    get_target = lambda data: (data['T'], data['E'])
    y_train = labtrans.transform(*get_target(train_dict_dh))
    y_valid = labtrans.transform(*get_target(valid_dict_dh))
    out_features = len(labtrans.cuts)
    duration_index = labtrans.cuts
    train_data = {'X': train_dict_dh['X'], 'T': y_train[0], 'E': y_train[1]}
    valid_data = {'X': valid_dict_dh['X'], 'T': y_valid[0], 'E': y_valid[1]}
    return train_data, valid_data, out_features, duration_index