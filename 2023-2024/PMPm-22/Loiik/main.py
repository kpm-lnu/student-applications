from generator import generate
from sir_model import calc_SIR
from seir_model import calc_SEIR
import comparator as comp


print('Generating data...')
data = generate()
print('Successfully generated.')

# print('Calculating SEIR_model...')
# SEIR_model_res = calc_SEIR()
SIR_avg = [0.063, 0.008]
SEIR_avg = [0.124, 0.3063, 0.9998, 0.3126, 0.01]
# SEIR_norm = norm(data, SEIR_model_res)
print('Fitting parameters...')
params = comp.parameter_estimation(data)

# print(params)

# print('Calculating SIR_model with average parameters...')
avg_SIR_model_res = calc_SIR(SIR_avg)
# print('Calculating SIR_model with fited parameters...')
fited_SIR_model_res = calc_SIR(params)

for i in range(3):
    for j in range(len(avg_SIR_model_res[0])):
        avg_SIR_model_res[i][j] *= 1000
        fited_SIR_model_res[i][j] *= 1000
#print(avg_SIR_model_res)
#print(fited_SIR_model_res)
#exit(0)


print('Norm1: ')
SIR_norm = comp.norm(data, avg_SIR_model_res)
print(SIR_norm)


print('Norm2: ')
SIR_norm = comp.norm(data, fited_SIR_model_res)
print(SIR_norm)

print(comp.norm(fited_SIR_model_res, avg_SIR_model_res))
print()
