from itertools import product

SCRATCH = '/scratch/snx3000/mboden'
# SCRATCH = '/scratch/mboden'

n_proc = 12 

default_params = {
'N':['256 256 256'],
'forcing_mode':['constant_eps'],
'init_mode':['Lamorgese'],
'save_path':['../results/test/'],
}

run_params = {
'Re_lam':[100],
'dt_ratio':[1],
'run':[3,4]
}

greasy_filename = 'run_test.txt'
case_script = 'Isotropic_transient.py'

# List of all hyper parameter combinations (list of dictionaries) to run
params_dict = {**default_params,**run_params}
hyper_params_dictionary_list = [dict(zip(params_dict.keys(),v)) for v in product(*params_dict.values())]
print('Number of hyperparameter combinations: {}'.format(len(hyper_params_dictionary_list)))
print('/!\\ check --nodes in bash script /!\\')

exec_path=SCRATCH+'/spectralDNS/deep-les/spectralDNS/'
with open('./greasy_tasks/'+greasy_filename, 'w') as file:
    for hyper_param_dict_case in hyper_params_dictionary_list:
        command = '[@ {:} @] -n {:} python3 {}'.format(exec_path, n_proc, case_script)
        for key, value in hyper_param_dict_case.items():
            print(key,value)
            try:
                if type(eval(value)) is dict:
                    value = "'"+str(value)+"'" 
            except:
                print('')

            command += ' --{:} {:}'.format(key, value)
        command+=' NS ;\n'
        file.write(command)




