import pickle

import joblib
import numpy

#with open('bonus_model.pkl', 'rb') as f:
#    clr = pickle.load(f)

fname = 'bonus_model.pkl'

# loaded_model = pickle.load(open(fname, 'rb'))
model = joblib.load(fname)

prediction = model['model'].predict_proba(numpy.array([[
1.02142857e+01,
0.00000000e+00,
1.00000000e+00,
0.00000000e+00,
4.39400000e+03,
0.00000000e+00,
6.50000000e+01,
2.40000000e+01,
1.00000000e+00,
4.12736808e-03,
1.88196259e-02,
9.58872274e+00,
9.46601299e-03,
1.62935551e+04,
3.05750000e+04,
8.00000000e+01,
2.30000000e+02,
6.20000000e+02,
1.86000000e+03,
2.48000000e+03,
7.30000000e+02,
1.55000000e+04,
7.85000000e+02,
2.07500000e+03,
1.00000000e+01,
5.18500000e+03,
4.75000000e+02,
2.80500000e+03,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00,
1.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00,
1.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
0.00000000e+00,
1.00000000e+00]]))

from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType

# initial_type = [('float_input', FloatTensorType([None, 98]))]
# onx = convert_sklearn(clr, initial_types=initial_type)
onx = to_onnx(model, X_train[:1])

with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

import onnxruntime as rt

sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

print(pred_onx)