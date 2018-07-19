. preprocess_new.sh config.json
. preprocess_with_cached.sh config.json
./preprocess_lu_bw.py config.json ./lu_bw_data.xlsx
./description_module.py config.json
wc -l ./env/data_files/*
wc -l ./env/data_files/lu_bw/*
./loader_module.py config.json
. save_final_dfs.sh
DF=$(grep -e '\"final_df_name\": \"\K(\S+)(?=.csv\")' config.json -oP | grep -e '_\K\d+\S+' -oP)
. archive_data_files.sh "./env/data_files_${DF}.tar.gz"
