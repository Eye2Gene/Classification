import os
import numpy as np
import sys

import tensorflow as tf
#tf.compat.v1.enable_eager_execution() 

#Hack to get relative import working
sys.path.insert(1, os.path.join(sys.path[0], '..'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('model', help='Path to model h5 file')
    parser.add_argument('--npy', help='Save a Numpy array with columns for the individual class predictions at the specified path')
    parser.add_argument('--csv', help='Save a CSV with columns for the individual class predictions at the specified path')
    parser.add_argument('--embedding', help='Store the raw embeddings before the final layer at the specified path')
    parser.add_argument('--no-softmax', action='store_true')
    parser.add_argument('--gpu', type=str, default="0")
    
    args = parser.parse_args()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model_path = args.model
    data_path = args.image_dir

    print('Loading model from ', model_path)
    from models import load_model
    model = load_model(model_path)

    print('Model loaded')
    print(model._config)

    if args.no_softmax:
        model.model.layers[-1].activation = None

    print('## Evaluating on test data ##')
    prediction, labels, filenames = model.predict(data_path,
                                                  return_labels=True,
                                                  return_filenames=True)
    
    pred_class = prediction.argmax(axis=-1)
    correct = (pred_class == labels)
    total = len(labels)

    print('Accuracy: {:.2f}, {}/{}'.format(correct.mean() * 100, correct.sum(), total))

    if args.npy:
        np.save(args.npy, {'file': filenames, 'prediction': prediction, 'true': labels, 'classes': sorted(model.classes)})

    if args.csv:
        if not data_path[-4:] == ".csv":
            raise Exception("Currently output to csv only supported whe using input csv")

        import pandas as pd
        df = pd.read_csv(data_path)
        path_col = model._config['dataseries_path']
        
        df_pred = pd.DataFrame()
        df_pred[path_col] = filenames
        df_pred["pred_class"] = [ sorted(model.classes)[i] for i in pred_class ] 
        for i, cls in enumerate(sorted(model.classes)):
            df_pred["pred_"+cls] = prediction[:,i]
        df_pred["pred_model"] = model_path
        
        df = pd.merge(df, df_pred, how='right', on=path_col)
        df.to_csv(args.csv, index=False)

    if args.embedding:
        # A hack to get embeddings quickly
        intermediate_layer_model = tf.keras.Model(inputs=model.model.input,
                                     outputs=model.model.layers[-2].output)
        model.model = intermediate_layer_model
        intermediate_output = model.predict(data_path)
        np.save(args.embedding, intermediate_output)
