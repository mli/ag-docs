#!/bin/bash

# cp -r ~/Google\ Drive/My\ Drive/Colab\ Notebooks/autogluon_tutorials/* .
# rm -rf _build/
rm -rf jupyter_execute/
sphinx-build -b html . _build/
