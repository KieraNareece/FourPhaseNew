# Import libraries to use in the following code
import types
import pandas as pd
from Functions import *
import time
import tkinter as tk

# Define time and parameters
days = 9
minutes_fermentation = days*24*60 + 60.   # Calculates number of minutes that the fermentation runs for
ts = np.arange(0, minutes_fermentation)     # Define time space
V_MS = 20                                  # L, volume of tank
V_GR = 0.25*V_MS                           # Approximate volume of grapes
V_CL = 0.1*V_MS                           # Approximate volume of grapes

start = time.time()

# # Implement Simple Name Space for the parameters of the system
# The units and the definitions are defined below

p = types.SimpleNamespace()

p.param = types.SimpleNamespace(k_d=9.660648089497706e-06, K_S=9.843439862211241, K_S2=149.07302856445312, K_N=0.01,
                                K_E=1, u_max=0.0007970177076081221, Y_LX=0.00001, Y_NX=0.032, Y_SX=0.16000000000000064,
                                Y_CX=0.56, Y_EX=1, Y_GX=0.56, Y_SCX=0.56, k_pp=5.500e-06, PI_LSPP=0.4, PI_GRA=1.6,
                                PI_LSA=5, PI_CLA=1, PI_GRT=0.3, PI_LST=1, PI_CLT=1, PI_GRTA=0.6, PI_LSTA=1, PI_CLTA=1,
                                PI_GRMA=0.4, PI_LSMA=1, PI_CLMA=1, a_CD=0.03, b_CD=0.02, a_TPI1=0.03, b_TPI1=0.02,
                                c_TPI1=0.02, a_TPI2=0.01, b_TPI2=0.02, kla_ml_pp=0.01, kla_mg_a_i=0.0004,
                                kla_df_a=0.004, kla_ml_a=0.08, kla_mg_t_i=0.002, kla_df_t=0.004, kla_ml_t=0.004,
                                kla_mg_ta_i=0.0001, kla_df_ta=0.004, kla_ml_ta=0.0004, kla_mg_ma_i=0.00007,
                                kla_df_ma=0.004, kla_ml_ma=0.0004, ea=0.002, et=0.0008460846885103096, eta=0.0001,
                                ema=0.000002, kla_max_a=0.021010790130807933, kla_max_t=0.01,
                                kla_max_ma=0.011010790130807933, kla_max_ta=0.011010790130807933,
                                B=0.0024352033212567746, Y_ES=3.441094938937915, Y_GS=0.003, Y_SCS=0.00001)

p.param_names = ['k_d', 'K_S', 'K_S2', 'K_N', 'K_E', 'u_max', 'Y_LX', 'Y_NX', 'Y_SX', 'Y_CX', 'Y_EX', 'Y_GX', 'Y_SCX',
                 'k_pp', 'PI_LSPP', 'PI_GRA', 'PI_LSA', 'PI_CLA', 'PI_GRT', 'PI_LST', 'PI_CLT', 'PI_GRTA', 'PI_LSTA',
                 'PI_CLTA', 'PI_GRMA', 'PI_LSMA', 'PI_CLMA', 'a_CD', 'b_CD', 'a_TPI1', 'b_TPI1', 'c_TPI1', 'a_TPI2',
                 'b_TPI2', 'kla_ml_pp', 'kla_mg_a_i', 'kla_df_a', 'kla_ml_a', 'kla_mg_t_i', 'kla_df_t', 'kla_ml_t',
                 'kla_mg_ta_i', 'kla_df_ta', 'kla_ml_ta', 'kla_mg_ma_i', 'kla_df_ma', 'kla_ml_ma', 'ea', 'et', 'eta',
                 'ema', 'kla_max_a', 'kla_max_t', 'kla_max_ma', 'kla_max_ta', 'B', 'Y_ES', 'Y_GS', 'Y_SCS']

p.SM = types.SimpleNamespace(X=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]),
                             N=np.array([0, -p.param.Y_NX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             S=np.array([-p.param.Y_ES*p.param.B, -p.param.Y_SX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CO2=np.array([p.param.B, p.param.Y_CX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             E=np.array([p.param.B, p.param.Y_EX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             GL=np.array([p.param.Y_GS, p.param.Y_GX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             SC=np.array([p.param.Y_SCS, p.param.Y_SCX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CPP=np.array([0, 0, -1/V_MS, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             MPP_LS=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CA=np.array([0, 0, 0, -1, 1/V_MS, -1/V_MS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CA_CL=np.array([0, 0, 0, 0, -1/V_CL, 0, 1/V_CL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CA_GR=np.array([0, 0, 0, 0, 0, 0, -1/V_GR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             MA_LS=np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CT=np.array([0, 0, 0, 0, 0, 0, 0, 1/V_MS, -1/V_MS, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CT_CL=np.array([0, 0, 0, 0, 0, 0, 0, -1/V_CL, 0, 1/V_CL, 0, 0, 0, 0, 0, 0, 0]),
                             CT_GR=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1/V_GR, 0, 0, 0, 0, 0, 0, 0]),
                             MT_LS=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                             CTA=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/V_MS, -1/V_MS, 0, 0, 0, 0, 0]),
                             CTA_CL=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/V_CL, 0, 1/V_CL, 0, 0, 0, 0]),
                             CTA_GR=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/V_GR, 0, 0, 0, 0]),
                             MTA_LS=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                             CMA=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/V_MS, -1/V_MS, 0, 0]),
                             CMA_CL=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/V_CL, 0, 1/V_CL, 0]),
                             CMA_GR=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/V_GR, 0]),
                             MMA_LS=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                             M_LS=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p.param.Y_LX*V_MS]))

p.svar_list = ['X', 'N', 'S', 'CO2', 'E', 'GL', 'SC', 'M_LS', 'CPP', 'MPP_LS', 'CA', 'CA_CL', 'CA_GR', 'MA_LS', 'CT',
               'CT_CL', 'CT_GR', 'MT_LS', 'CTA', 'CTA_CL', 'CTA_GR', 'MTA_LS', 'CMA', 'CMA_CL', 'CMA_GR', 'MMA_LS']
p.comb_list = ['X', 'N', 'S', 'CO2', 'E', 'GL', 'SC', 'M_LS', 'CPP', 'MPP_LS', 'CA', 'CA_CL', 'CA_GR', 'MA_LS', 'CT',
               'CT_CL', 'CT_GR', 'MT_LS', 'CTA', 'CTA_CL', 'CTA_GR', 'MTA_LS', 'CMA', 'CMA_CL', 'CMA_GR', 'MMA_LS']

p.scenario = ['All', 'Cultivar', 'Temperature', 'Specific']
p.cultivars = ['CS', 'SH', 'ME']
p.temperature = ['20', '25', '28']
p.cap_components = ['Anthocyanins', 'Tannins', 'Total_Phenolic_Index', 'Tartaric_Acid', 'Malic_Acid']
p.lees_components = ['Anthocyanins', 'Polymeric_Pigments', 'Tannins', 'Tartaric_Acid', 'Malic_Acid']
p.must_components = ['Anthocyanins', 'Polymeric_Pigments', 'Tannins', 'Total_Phenolic_Index', 'Tartaric_Acid',
                     'Malic_Acid', 'Colour_Density', 'Succinic_Acid', 'Biomass', 'Sugar', 'Ethanol', 'pH', 'CO2',
                     'Nitrogen']
p.repeat = ['Repeat_1', 'Repeat_2', 'Repeat_3']
p.phase_name = ['Must', 'Cap', 'Lees', 'Grape']
p.interest = ['X', 'N', 'S', 'CO2', 'E', 'SC', 'CPP', 'CA', 'CT', 'CTA', 'CMA', 'cd', 'tpi', 'pH_must']
p.exp_components = ['Biomass', 'Nitrogen', 'Sugar', 'CO2', 'Ethanol', 'Succinic_Acid', 'Polymeric_Pigments',
                    'Anthocyanins', 'Tannins', 'Tartaric_Acid', 'Malic_Acid', 'Colour_Density', 'Total_Phenolic_Index',
                    'pH']

reference_comp = ['X', 'X', 'X', 'M_LS', 'N', 'N', 'N', 'N', 'S', 'S', 'S', 'S', 'CO2', 'CO2', 'CO2', 'CO2', 'E', 'E',
                  'E', 'E', 'GL', 'GL', 'GL', 'GL', 'SC', 'SC', 'SC', 'SC', 'CPP', 'CPP', 'CPP', 'MPP_LS', 'CA',
                  'CA_CL', 'CA_GR', 'MA_LS', 'CT', 'CT_CL', 'CT_GR', 'MT_LS', 'CTA', 'CTA_CL', 'CTA_GR', 'MTA_LS',
                  'CMA', 'CMA_CL', 'CMA_GR', 'MMA_LS', 'cd', 'cd', 'cd', 'cd', 'tpi_gr', 'tpi', 'tpi', 'tpi',
                  'pH_must', 'ph_must', 'pH_must', 'pH_must']
input_ref = ['1X', '2X', '3X', '4X', '1N', '2N', '3N', '4N', '1S', '2S', '3S', '4S', '1CO2', '2CO2', '3CO2', '4CO2',
             '1ET', '2ET', '3ET', '4ET', '1GL', '2GL', '3GL', '4GL', '1SC', '2SC', '3SC', '4SC', '1PP', '2PP', '3PP',
             '4PP', '3A', '2A', '1A', '4A', '3T', '2T', '1T', '4T', '3TA', '2TA', '1TA', '4TA', '3MA', '2MA', '1MA',
             '4MA', '1CD', '2CD', '3CD', '4CD', '1TPI', '2TPI', '3TPI', '4TPI', '1P', '2P', '3P', '4P']

reference_comp_average = ['Must_20_Biomass', 'Must_20_Biomass', 'Must_20_Biomass', 'Must_20_Biomass',
                          'Must_20_Nitrogen', 'Must_20_Nitrogen', 'Must_20_Nitrogen', 'Must_20_Nitrogen',
                          'Must_20_Sugar', 'Must_20_Sugar', 'Must_20_Sugar', 'Must_20_Sugar', 'Must_20_CO2',
                          'Must_20_CO2', 'Must_20_CO2', 'Must_20_CO2', 'Must_20_Ethanol', 'Must_20_Ethanol',
                          'Must_20_Ethanol', 'Must_20_Ethanol', 'Must_20_Succinic_Acid', 'Must_20_Succinic_Acid',
                          'Must_20_Succinic_Acid', 'Must_20_Succinic_Acid', 'Must_20_Polymeric_Pigments',
                          'Must_20_Polymeric_Pigments', 'Must_20_Polymeric_Pigments', 'Lees_20_Polymeric_Pigments',
                          'Must_20_Anthocyanins', 'Cap_20_Anthocyanins', 'Must_20_Anthocyanins', 'Lees_20_Anthocyanins',
                          'Must_20_Tannins', 'Cap_20_Tannins', 'Must_20_Tannins', 'Lees_20_Tannins',
                          'Must_20_Tartaric_Acid', 'Cap_20_Tartaric_Acid', 'Must_20_Tartaric_Acid',
                          'Lees_20_Tartaric_Acid', 'Must_20_Malic_Acid', 'Cap_20_Malic_Acid', 'Must_20_Malic_Acid',
                          'Lees_20_Malic_Acid', 'Must_20_Colour_Density', 'Must_20_Colour_Density',
                          'Must_20_Colour_Density', 'Must_20_Colour_Density', 'Cap_20_Total_Phenolic_Index',
                          'Must_20_Total_Phenolic_Index', 'Must_20_Total_Phenolic_Index',
                          'Must_20_Total_Phenolic_Index', 'Must_20_pH', 'Must_20_pH', 'Must_20_pH', 'Must_20_pH']
input_ref_average = ['1X', '2X', '3X', '4X', '1N', '2N', '3N', '4N', '1S', '2S', '3S', '4S', '1CO2', '2CO2', '3CO2',
                     '4CO2', '1ET', '2ET', '3ET', '4ET', '1SC', '2SC', '3SC', '4SC', '1PP',
                     '2PP', '3PP', '4PP', '3A', '1A', '2A', '4A', '3T', '1T', '2T', '4T', '3TA', '1TA', '2TA', '4TA',
                     '3MA', '1MA',  '2MA', '4MA', '1CD', '2CD', '3CD', '4CD', '1TPI', '3TPI', '3TPI', '4TPI', '1P',
                     '2P', '3P', '4P']

index_dictionary = {input_ref[i]: reference_comp[i] for i in range(len(input_ref))}
index_dictionary_average = {input_ref_average[i]: reference_comp_average[i] for i in range(len(input_ref_average))}

df_must = pd.read_csv('D:\\Stellenbosch\\2. PhD\\2022\\0. Experimental\\Data Sets\\wine_data_N.csv')
df_lees = pd.read_csv('D:\\Stellenbosch\\2. PhD\\2022\\0. Experimental\\Data Sets\\lees_data.csv')
df_cap = pd.read_csv('D:\\Stellenbosch\\2. PhD\\2022\\0. Experimental\\Data Sets\\cap_data.csv')
df_grape = pd.read_csv('D:\\Stellenbosch\\2. PhD\\2022\\0. Experimental\\Data Sets\\grape_data.csv')

df = types.SimpleNamespace(Must=df_must, Lees=df_lees, Cap=df_cap, Grape=df_grape)
phase_name = ['Must', 'Cap', 'Lees', 'Grape']

data, list_values = experimental_data(df, phase_name)
sd, average = standard_deviations(df, phase_name, p)

x = types.SimpleNamespace(X=0.25, N=average.Must_20_Nitrogen[0], S=average.Must_20_Sugar[0], CO2=0, E=0, GL=0, SC=0.015,
                          M_LS=0.001, CPP=average.Must_20_Polymeric_Pigments[0], MPP_LS=0.00,
                          CA=average.Must_20_Anthocyanins[0], CA_CL=average.Must_20_Anthocyanins[0], CA_GR=1009.6,
                          MA_LS=0, CT=average.Must_20_Tannins[0], CT_CL=average.Must_20_Tannins[0],
                          CT_GR=5384.49, MT_LS=0, CTA=average.Must_20_Tartaric_Acid[0],
                          CTA_CL=average.Must_20_Tartaric_Acid[0], CTA_GR=3, MTA_LS=0,
                          CMA=average.Must_20_Malic_Acid[0], CMA_CL=average.Must_20_Malic_Acid[0], CMA_GR=2.11,
                          MMA_LS=0)


time_points = (df_must['Hour_Must']).to_numpy()
time_array = time_points*60
time_array[10] = 13019

start = time.time()
sol = speed_up(p, x)
solution = vector_to_namespace(sol, p.comb_list)
v = simple_calc(ts, solution, p, time_array)
stop = time.time()
print(stop-start)
# gs = p.param
# gl = p.param_names
# g = namespace_to_vector(gs, gl)
# case = ['n/a', '20']
# exp_vector, stdev_vector, determinate = residuals(case, data, time_array, sd, p)
#
#
# lp = types.SimpleNamespace(k_d=9.5e-06, K_S=9.7, K_S2=145, K_N=0.01, K_E=0.8, u_max=0.0007, Y_LX=0.000005, Y_NX=0.030,
#                            Y_SX=0.15, Y_CX=0.5, Y_EX=0.5, Y_GX=0.55, Y_SCX=0.55, k_pp=5.00e-06, PI_LSPP=0.0,
#                            PI_GRA=0.5, PI_LSA=1, PI_GRT=0.3, PI_LST=0, PI_GRTA=0, PI_LSTA=0, PI_GRMA=0, PI_LSMA=0,
#                            a_CD=0.01, b_CD=0.01, a_TPI1=0.01, b_TPI1=0.01, c_TPI1=0.01, a_TPI2=0.005, b_TPI2=0.01,
#                            kla_ml_pp=0.005, kla_mg_a_i=0.0002, kla_ml_a=0.05, kla_mg_t_i=0.001, kla_ml_t=0.001,
#                            kla_mg_ta_i=0.00005, kla_ml_ta=0.0005, kla_mg_ma_i=0.00005, kla_ml_ma=0.0001, ea=0.001,
#                            et=0.0005, eta=0.00005, ema=0.0000015, kla_max_a=0.01, kla_max_t=0.005, kla_max_ma=0.005,
#                            kla_max_ta=0.005, B=0.001, Y_ES=3., Y_GS=0.001, Y_SCS=0.000005)
# #
# up = types.SimpleNamespace(k_d=9.7e-06, K_S=10, K_S2=155, K_N=0.03, K_E=1.2, u_max=0.0009, Y_LX=0.000015, Y_NX=0.035,
#                            Y_SX=0.2, Y_CX=0.6, Y_EX=1.5, Y_GX=0.6, Y_SCX=0.6, k_pp=6.000e-06, PI_LSPP=0.5,
#                            PI_GRA=2, PI_LSA=5, PI_GRT=0.5, PI_LST=1, PI_GRTA=0.6, PI_LSTA=1, PI_GRMA=0.4, PI_LSMA=1,
#                            a_CD=0.05, b_CD=0.05, a_TPI1=0.05, b_TPI1=0.05, c_TPI1=0.05, a_TPI2=0.015, b_TPI2=0.05,
#                            kla_ml_pp=0.015, kla_mg_a_i=0.0005, kla_ml_a=0.1, kla_mg_t_i=0.005, kla_ml_t=0.005,
#                            kla_mg_ta_i=0.0005, kla_ml_ta=0.001, kla_mg_ma_i=0.0001, kla_ml_ma=0.0005, ea=0.005,
#                            et=0.001, eta=0.00015, ema=0.0000025, kla_max_a=0.05, kla_max_t=0.01, kla_max_ma=0.015,
#                            kla_max_ta=0.015, B=0.0025, Y_ES=4, Y_GS=0.005, Y_SCS=0.000015)
#
# lower_bounds = namespace_to_vector(lp, p.param_names)
# upper_bounds = namespace_to_vector(up, p.param_names)
# bounds = (lower_bounds, upper_bounds)
#
# a = least_squares(objective, g, method='trf', bounds=bounds, args=(p, x, ts, time_array, gl, determinate, exp_vector,
#                                                                    stdev_vector), verbose=2)
# print(a)
# new_p = vector_to_namespace(a, p.param_names)
# print(new_p)
# p.param = a
sol2 = speed_up(p, x)
solution2 = vector_to_namespace(sol2, p.comb_list)
v2 = simple_calc(ts, solution2, p, time_array)
# stop = time.time()
#
# print(stop-start)
#
ts = np.arange(0, minutes_fermentation)


root = tk.Tk()


def display(time_waiting):

    if time_waiting == 1:
        canvas1 = tk.Canvas(root, width=700, height=700)
        canvas1.pack()

        label1 = tk.Label(root, text='View Graph for Component')
        label1.config(font=('helvetica', 14))
        canvas1.create_window(300, 25, window=label1)

        label2 = tk.Label(root, text='Phase')
        label2.config(font=('helvetica', 10))
        canvas1.create_window(200, 120, window=label2)

        label3 = tk.Label(root, text='Component')
        label3.config(font=('helvetica', 10))
        canvas1.create_window(400, 120, window=label3)

        label4 = tk.Label(root, text='Component Codes:')
        label4.config(font=('helvetica', 10))
        canvas1.create_window(450, 180, window=label4)

        label5 = tk.Label(root, text='A - Anthocaynins')
        label5.config(font=('helvetica', 10))
        canvas1.create_window(450, 200, window=label5)

        label6 = tk.Label(root, text='CD - Colour Density')
        label6.config(font=('helvetica', 10))
        canvas1.create_window(455, 220, window=label6)

        label7 = tk.Label(root, text='PP - Polymeric Pigments')
        label7.config(font=('helvetica', 10))
        canvas1.create_window(472, 240, window=label7)

        label8 = tk.Label(root, text='T - Tannins')
        label8.config(font=('helvetica', 10))
        canvas1.create_window(432, 260, window=label8)

        label9 = tk.Label(root, text='TPI - Total Phenolic Index')
        label9.config(font=('helvetica', 10))
        canvas1.create_window(472, 280, window=label9)

        label10 = tk.Label(root, text='ET - Alcohol')
        label10.config(font=('helvetica', 10))
        canvas1.create_window(435, 300, window=label10)

        label11 = tk.Label(root, text='X-Biomass')
        label11.config(font=('helvetica', 10))
        canvas1.create_window(435, 320, window=label11)

        label12 = tk.Label(root, text='GL - Glycerol')
        label12.config(font=('helvetica', 10))
        canvas1.create_window(435, 340, window=label12)

        label13 = tk.Label(root, text='MA - Malic Acid')
        label13.config(font=('helvetica', 10))
        canvas1.create_window(445, 360, window=label13)

        label14 = tk.Label(root, text='TA - Tartaric Acid')
        label14.config(font=('helvetica', 10))
        canvas1.create_window(445, 380, window=label14)

        label15 = tk.Label(root, text='S - Total Sugar')
        label15.config(font=('helvetica', 10))
        canvas1.create_window(445, 400, window=label15)

        label16 = tk.Label(root, text='P - pH')
        label16.config(font=('helvetica', 10))
        canvas1.create_window(445, 480, window=label16)

        label17 = tk.Label(root, text='N - Nitrogen')
        label17.config(font=('helvetica', 10))
        canvas1.create_window(445, 420, window=label17)

        label18 = tk.Label(root, text='CO2 - Carbon Dioxide')
        label18.config(font=('helvetica', 10))
        canvas1.create_window(445, 440, window=label18)

        label19 = tk.Label(root, text='SC - Succinic Acid')
        label19.config(font=('helvetica', 10))
        canvas1.create_window(445, 460, window=label19)

        label25 = tk.Label(root, text='1 - Grape/Cap')
        label25.config(font=('helvetica', 10))
        canvas1.create_window(200, 200, window=label25)

        label26 = tk.Label(root, text='2 - Cap Liquid')
        label26.config(font=('helvetica', 10))
        canvas1.create_window(200, 220, window=label26)

        label27 = tk.Label(root, text='3 - Must')
        label27.config(font=('helvetica', 10))
        canvas1.create_window(200, 240, window=label27)

        label28 = tk.Label(root, text='4 - Lees')
        label28.config(font=('helvetica', 10))
        canvas1.create_window(200, 260, window=label28)

        entry1 = tk.Entry(root)
        canvas1.create_window(200, 140, window=entry1)

        entry2 = tk.Entry(root)
        canvas1.create_window(400, 140, window=entry2)

        def getgraph():

            p_num = entry1.get()
            a = entry2.get()

            graph(p_num, a, ts, solution, v, solution2, v2, index_dictionary, average, index_dictionary_average,
                  time_array)

        button1 = tk.Button(text='Get Graph', command=getgraph)
        canvas1.create_window(300, 180, window=button1)

    else:
        root.destroy()

    root.after(7185000, display, time_waiting - 1)


display(1)
root.mainloop()
