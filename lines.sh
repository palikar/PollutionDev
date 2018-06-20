for f in $(ls env/data_files/*.csv)
do
    wc -l $f
done
