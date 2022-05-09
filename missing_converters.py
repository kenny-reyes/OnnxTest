import joblib
import numpy
from numpy.testing import assert_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import onnxruntime
from skl2onnx.common.data_types import guess_data_type
from skl2onnx.common.exceptions import MissingShapeCalculator
from skl2onnx.helpers import collect_intermediate_steps, compare_objects, enumerate_pipeline_models
from skl2onnx.helpers.investigate import _alter_model_for_debugging
from skl2onnx import convert_sklearn

class MyScaler(StandardScaler):
    pass

# Let's fit a model.
data = numpy.array([[0, 0], [0, 0], [2, 1], [2, 1]],
                   dtype=numpy.float32)
model = Pipeline([("scaler1", StandardScaler()),
                  ("scaler2", StandardScaler()),
                  ("scaler3", MyScaler()),
                ])
model.fit(data)

fname = 'bonus_model.pkl'
model = joblib.load(fname)
model = model['model']
# This function alters the pipeline, every time
# methods transform or predict are used, inputs and outputs
# are stored in every operator.
_alter_model_for_debugging(model, recursive=True)

# Let's use the pipeline and keep intermediate
# inputs and outputs.
# model.transform(data)

# Let's get the list of all operators to convert
# and independently process them.
all_models = list(enumerate_pipeline_models(model))

# Loop on every operator.
for ind, op, last in all_models:
    if ind == (0,):
        # whole pipeline
        continue

    # The dump input data for this operator.
    data_in = op._debug.inputs['transform']

    # Let's infer some initial shape.
    t = guess_data_type(data_in)

    # Let's convert.
    try:
        onnx_step = convert_sklearn(op, initial_types=t)
    except MissingShapeCalculator as e:
        if "MyScaler" in str(e):
            print(e)
            continue
        raise

    # If it does not fail, let's compare the ONNX outputs with
    # the original operator.
    sess = onnxruntime.InferenceSession(onnx_step.SerializeToString())
    onnx_outputs = sess.run(None, {'input': data_in})
    onnx_output = onnx_outputs[0]
    skl_outputs = op._debug.outputs['transform']
    assert_almost_equal(onnx_output, skl_outputs)
    compare_objects(onnx_output, skl_outputs)