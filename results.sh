#!/usr/bin/env bash


./gen_results.py ~/core.d/thesis_results/1d_res* --dest  ~/core.d/thesis_results/figs_1d/ --table-tex --basic-plots --feature-imp
./gen_results.py ~/core.d/thesis_results/12h_res* --dest  ~/core.d/thesis_results/figs_12h/ --table-tex --basic-plots --feature-imp
./gen_results.py ~/core.d/thesis_results/1h_res* --dest  ~/core.d/thesis_results/figs_1h/ --table-tex --basic-plots --feature-imp
