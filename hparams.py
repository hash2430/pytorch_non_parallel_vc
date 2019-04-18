import yaml

def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict

def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user

class Dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        if not dct:
            dct = dict()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value

class Hparam(Dotdict):
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    def __init__(self):
        super(Dotdict, self).__init__()

    def set_hparam_yaml(self, exp_name, default_file='configs/default.yaml'):
        #default_hp = load_hparam(default_file)
        exp_file = 'configs/{}.yaml'.format(exp_name)
        exp_hp = load_hparam(exp_file)
        #merge_dic = merge_dict(exp_hp, default_hp)
        hp_dict = Dotdict(exp_hp)
        for k, v in hp_dict.items():
            setattr(self, k, v)
        self._auto_setting(exp_name)

    def _auto_setting(self, exp_name):
        setattr(self, 'exp_name', exp_name)
        setattr(self, 'logdir', './logdir/{}'.format(exp_name))

hparam = Hparam()

