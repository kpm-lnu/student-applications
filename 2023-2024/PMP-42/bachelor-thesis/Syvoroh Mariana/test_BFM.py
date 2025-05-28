import numpy as np
import scipy.io as sio
from pathlib import Path

def load_bfm_model(model_path):
    bfm = sio.loadmat(model_path)
    model = bfm['model']
    model = model[0, 0]
    return model

model_path = Path("01_MorphableModel.mat")
bfm_model = load_bfm_model(model_path)

print("Model loaded successfully")
def generate_face(model, shape_params, expression_params):
    shape_model = model['shapeMU'] + model['shapePC'] @ shape_params
    expression_model = model['expMU'] + model['expPC'] @ expression_params
    face_model = shape_model + expression_model
    return face_model

# Приклад параметрів
shape_params = np.random.randn(199)  # Змінити відповідно до розміру вашої моделі
expression_params = np.random.randn(29)  # Змінити відповідно до розміру вашої моделі

face_model = generate_face(bfm_model, shape_params, expression_params)

print("Face model generated successfully")
