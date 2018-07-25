datasets = {'a1a', 'colon-cancer', 'covtype_binary_scale', 'australian_scale', 'breast_cancer_scale', 'usps_3vs5'};

for dataset_name = datasets
	seed  = 1;
	[y, X, y_te, X_te] = get_data_log_reg(dataset_name{1}, seed);
	file_name = strcat('./datasets/', dataset_name{1});
	save(file_name, 'dataset_name', 'y', 'X', 'y_te', 'X_te');
end
