from itertools import product

SCRATCH = '/scratch/snx3000/mboden'
n_proc = 6 

default_params = {
'N':['256 256 256'],
'forcing_mode':['constant_eps'],
'init_mode':['Lamorgese'],
}

run_params = {
'Re_lam':[60,80,100,120,140],
'dt_ratio':[1,2,5,10],
'run':[1,2,3]
}

# List of all hyper parameter combinations (list of dictionaries) to run
params_dict = {**default_params,**run_params}
hyper_params_dictionary_list = [dict(zip(params_dict.keys(),v)) for v in product(*params_dict.values())]
print('Number of hyperparameter combinations: {}'.format(len(hyper_params_dictionary_list)))
print('/!\\ check --nodes in bash script /!\\')

exec_path=SCRATCH+'/spectralDNS/deep-les/spectralDNS/'
model_name='md_arnn'
with open('./greasy_tasks/runDNStransient.txt', 'w') as file:
    for hyper_param_dict_case in hyper_params_dictionary_list:
        command = '[@ {:} @] -n {:} python3 Isotropic_transient.py'.format(exec_path, n_proc)
        for key, value in hyper_param_dict_case.items():
            try:
                if type(eval(value)) is dict:
                    value = "'"+str(value)+"'" 
            except:
                print(' ')

            command += ' --{:} {:}'.format(key, value)
        command+=' NS ;\n'
        file.write(command)




