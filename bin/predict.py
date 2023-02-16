import os
import numpy as np
import sys

import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

#Hack to get relative import working
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.base import Model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('model')
    parser.add_argument('--npy')
    parser.add_argument('--csv')
    parser.add_argument('--embedding')
    parser.add_argument('--no-softmax', action='store_true')

    args = parser.parse_args()

    model_path = args.model
    data_path = args.image_dir

    print('Loading model from ', model_path)
    model = Model().load(model_path)

    print('Model loaded')
    print(model._config)

    if args.no_softmax:
        model.model.layers[-1].activation = None

    print('## Evaluating on test data ##')
    prediction, labels = model.predict(data_path, return_labels=True)
    prediction_class = prediction.argmax(axis=-1)

    correct = (prediction_class == labels).sum()
    total = len(labels)

    print('Percentage correct (manual): {:.2f}, {}/{}'.format((correct / total * 100), correct, total))

    if args.npy:
        np.save(args.npy, {'prediction': prediction, 'true': labels, 'classes': sorted(model.classes)})

    if args.csv:
        if not data_path[-4:] == ".csv":
            raise Exception("Currently output to csv only supported whe using input csv")

        import pandas as pd
        df = pd.read_csv(data_path)
        df = df[ df[model._config['dataseries_label']].isin(model.classes) ]
        df["pred.class"] = [ sorted(model.classes)[i] for i in prediction_class ] 
        for i, cls in enumerate(sorted(model.classes)):
            df["pred."+cls] = prediction[:,i]
        df["pred.model"] = [ model_path ]*len(df)
        df.to_csv(args.csv, index=False)

    if args.embedding:
        # A hack to get embeddings quickly
        intermediate_layer_model = tf.keras.Model(inputs=model.model.input,
                                     outputs=model.model.layers[-2].output)
        model.model = intermediate_layer_model
        intermediate_output = model.predict(data_path)
        np.save(args.embedding, intermediate_output)
