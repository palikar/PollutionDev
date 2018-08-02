#!/usr/bin/bash
# Setup anaconda

. /etc/profile.d/anaconda.sh

setup-anaconda
source activate edward

# Now run the python script with the
# parameters defined in the HTCondor submit file
echo "Starting job ..."


./bnn_model.py $*

echo "Finishing job"


EXITCODE=$?

exit $EXITCODE
