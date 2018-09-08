#!/usr/bin/bash
. /etc/profile.d/anaconda.sh
setup-anaconda
source activate edward
echo "Starting job ..."

# mkdir -p "/smartdata/ujevj/${FOLDER_RES}"



FOLDER_RES="1d_res"

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_1" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_2" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1 --period 1D --outvalue P1 --dest   "/smartdata/ujevj/${FOLDER_RES}/mdn_train_3" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_4" --base-dir "./"

#

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_5" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_6" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_7" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_8" --base-dir "./"

#

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_9" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_10" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_11" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_12" --base-dir "./"


FOLDER_RES="12h_res"

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_1" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_2" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1 --period 12H --outvalue P1 --dest   "/smartdata/ujevj/${FOLDER_RES}/mdn_train_3" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_4" --base-dir "./"

#

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_5" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_6" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_7" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_8" --base-dir "./"

#

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_9" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_10" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_11" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_12" --base-dir "./"



FOLDER_RES="1h_res"

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_1" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_2" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P1 --period 1H --outvalue P1 --dest   "/smartdata/ujevj/${FOLDER_RES}/mdn_train_3" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SBC --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_4" --base-dir "./"

#

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_5" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_6" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_7" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SNTR --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_8" --base-dir "./"

#

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_9" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_10" --base-dir "./" --take_lubw

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_11" --base-dir "./"

python model_training.py --config ./model_config.json --model mdn --station SAKP --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_12" --base-dir "./"


FOLDER_RES="1d_res_sec"

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_1" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_2" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P1 --period 1D --outvalue P1 --dest   "/smartdata/ujevj/${FOLDER_RES}/mdn_train_3" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_4" --base-dir "./"

#

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_5" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_6" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_7" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_8" --base-dir "./"

#

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_9" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_10" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P1 --period 1D --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_11" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P2 --period 1D --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_12" --base-dir "./"


FOLDER_RES="12h_res_sec"

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_1" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_2" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P1 --period 12H --outvalue P1 --dest   "/smartdata/ujevj/${FOLDER_RES}/mdn_train_3" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_4" --base-dir "./"


#

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_5" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_6" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_7" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_8" --base-dir "./"

#

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_9" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_10" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P1 --period 12H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_11" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P2 --period 12H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_12" --base-dir "./"



FOLDER_RES="1h_res_sec"

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_1" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_2" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P1 --period 1H --outvalue P1 --dest   "/smartdata/ujevj/${FOLDER_RES}/mdn_train_3" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SBC --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_4" --base-dir "./"

#

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_5" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_6" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_7" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SNTR --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_8" --base-dir "./"

#

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_9" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_10" --base-dir "./" --take_lubw

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P1 --period 1H --outvalue P1 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_11" --base-dir "./"

python model_training.py --config ./model_config_2.json --model mdn --station SAKP --predictor P2 --period 1H --outvalue P2 --dest  "/smartdata/ujevj/${FOLDER_RES}/mdn_train_12" --base-dir "./"


#--load-mdn ./test_eval/mdn_model/model --load-bnn ./test_eval/bnn_model/ --take_lubw
#SBC
#SAKP - bad one
#SNTR


# P1, P2
# with, without LUBW
# station - SBC, SNTR

echo "Finishing job"
# EXITCODE=$?
exit $EXITCODE
