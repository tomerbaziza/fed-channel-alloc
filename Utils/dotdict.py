

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


# d = dotdict()
# d.yaniv = []
# d.ot = 43

# import pickle 

# with open("dfdf.pk", "wb") as f:
#     pickle.dump(d, f)