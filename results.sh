#!/usr/bin/env bash


./gen_results.py $1/1d_res* --dest  ./Thesis/figures/figs_1d/ --table-tex --basic-plots --feature-imp --pred-check
./gen_results.py $1/12h_res* --dest ./Thesis/figures/figs_12h/ --table-tex --basic-plots --feature-imp --pred-check
./gen_results.py $1/1h_res* --dest  ./Thesis/figures/figs_1h/ --table-tex --basic-plots --feature-imp --pred-check
