DF=$(grep -e '\"final_df_name\": \"\K(\S+)(?=.csv\")' config.json -oP | grep -e '_\K\d+\S+' -oP)
LOG="${DF}_log.txt"
touch $LOG
echo "Running preprocessesor for generating the cache" | tee -a $LOG
. preprocess_new_cache.sh config.json | tee -a  $LOG
echo "Running preprocessor for generating data files" | tee -a  $LOG
. preprocess_with_cached.sh config.json | tee -a  $LOG
echo "Processing the LU BW files" | tee -a  $LOG
./preprocess_lu_bw.py config.json ./lu_bw_data.xlsx | tee -a  $LOG
echo "Running the description module for final data processing" | tee -a  $LOG
./description_module.py config.json | tee -a  $LOG
echo "Lines in the luftdaten.de datafiles" | tee -a  $LOG
wc -l ./env/data_files/* | tee -a  $LOG
echo "Lines in the lu bw datafiles" | tee -a  $LOG
wc -l ./env/data_files/lu_bw/* | tee -a  $LOG
echo "Generating the final dataframe" | tee -a  $LOG
./loader_module.py config.json | tee -a  $LOG
echo "Copying dataframes to the right folder" | tee -a  $LOG
. save_final_dfs.sh | tee -a  $LOG
echo "Archiving relevent files" | tee -a  $LOG
sleep 5
. archive_data_files.sh "./env/data_files_${DF}.tar.gz" | tee -a  $LOG
echo "Done!"
