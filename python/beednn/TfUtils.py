import tensorflow as tf
import numpy as np
import json

def save_tf_to_json(model,filename):
    
    j={}
    i=0
    for l in model.layers:
        lj={}
        lj["name"]=l.name
        all_weights=l.weights
        wi=0
        for w in all_weights:
            ws=(w.numpy().tolist())
            lj["weight_"+str(wi)]=ws
            wi=wi+1

        if hasattr(l,"activation"):
            lj["activation"]=l.activation.__name__

        j["layer_"+str(i)]=lj
        i=i+1

    f=open(filename,'w')
    json.dump(j,f,indent=4)
    f.close()
