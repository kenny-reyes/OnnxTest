import joblib
import numpy
from sklearn.linear_model import LogisticRegression
import pickle

#with open('bonus_model.pkl', 'rb') as f:
#    clr = pickle.load(f)

fname = 'bonus_model.pkl'
clr = joblib.load(fname)

# clr = LogisticRegression(max_iter=1000)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

{'COTIZACION': 210070689, 'FECEQUIPO': Timestamp('2019-09-04 12:22:46'), 'ID': 19553.0, 'ID_MES': 201905.0, 'CODPOSTAL': 11540.0, 'FECPRIMERAMAT': '2006-01-01 00:00:00', 'ACCESORIOS': 0.0, 'MARCA': 'otro', 'MODELO': 5.0, 'VERSION': 1.0, 'TIPO_DE_DOCUMENTO': 'NIF', 'JURIDICOPROP': 'N', 'USO': 10, 'FNACIMIENTOCOND': '1970-01-01 00:00:00', 'MASCONDUCTORES': 0.0, 'FCARNETCOND': '1990-01-01 00:00:00', 'PETICION': 'ag00923w.TIREA16112021164707', 'SEGMENTACION': 'C1', 'TARA': 1324.420404567688, 'POTENCIA_CV': 114.17232336300086, 'TAMANO': 'G', 'PRECIO_VP': 60000.0, 'LONGITUD': 4190.146133012187, 'COMBUSTIBLE': 'D', 'CENTCUB': 1896.0, 'SEGMENTO_ESTRATEGICO_3112': 140.0, 'NOTA1': 'Z', 'NOTA2V': 3, 'NIVELBONUS': 1.0, 'PORCENTAJEBONUS': -58.0, 'ESMULTITARIFICADOR': 'S', 'PERFIL_USUARIO': 'SERVWEBA', 'PERFIL': 'SERVICIOSWEB', 'MEDIADOR': 30244.0, 'DESCRIPCIONCANAL': 'CORREDORES', 'CODIGO_SEGMENTO_MEDIADOR': '999', 'TIPO': 10, 'SEXOCOND': 'H', 'MISMAFIGURA': 1.0, 'ANTIGPOLIZA': 5.0, 'SINCOSINIESTROS': 1.0, 'PRIMA_TARIFICACION': 0.0, 'NANTVEHIC': 12.703346805515478, 'NEDADHAB': 51.14358921716335, 'NANTCARNETHAB': 32.0, 'NEDADEXPEDCARNETHAB': 22.34820197381969, 'COD_SEG_VALOR': 99.0, 'COD_SEG_VALOR3112': 999.0, 'IND_NUEVO_12M': 0.0, 'BMAX': 0.0, 'days_tmin_5': 0.0, 'days_tmax_30': 4.77027608364141, 'snow_soil_days': 0.0, 'average_income': 21331.49, 'total_population': 62070.0, 'num_acc': 493.0, 'imd_total': 27363.86740331492, 'illiterate_people_from_16_to_64_yo': 0.0, 'illiterate_people_over_64_yo': 0.0, 'people_wo_studies_from_16_to_64_yo': 0.0, 'people_wo_studies_over_64_yo': 0.0, 'people_w_1st_grade_studies_from_16_to_64_yo': 0.0, 'people_w_1st_grade_studies_over_64_yo': 0.0, 'people_w_2nd_grade_studies_from_16_to_64_yo': 22965.0, 'people_w_2nd_grade_studies_over_64_yo': 0.0, 'people_w_3rd_grade_studies_from_16_to_64_yo': 0.0, 'people_w_3rd_grade_studies_over_64_yo': 0.0, 'people_under_16_yo_wo_info_studies': 0.0, 'cel_type': 'urbano', 'illiterate_people': 0.0, 'people_wo_studies': 0.0, 'peso_potencia': 11.614644478406497, 'WEB': 1.0, 'top': 0.007035555214562898}

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

import onnxruntime as rt

sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

print(pred_onx)