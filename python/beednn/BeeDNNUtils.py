import numpy as np
import json
from . import Model,Layer

def load_from_json(filename):
    m=Model.Model()

    f=open(filename)
    j=json.load(f)
    f.close()

    for a in j:
        if 'layer' in str(a):
            l=j[a]
            if l['name']=='dense':
                lb=Layer.LayerDense()
                w_0=l['weight_0']
                w_1=l['weight_1']
                activation=l['activation']
#todo
            m.append(lb)
    return m