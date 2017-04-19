#!/bin/zsh

rm *.h5


for ID in M2{Id,Fou} M2_{01,02,03}; do
    export ID
    export BASEDIR="$HOME/Downloads/Phaselift_2017_13_3/2x2 phaselift"
    jupyter nbconvert --to html --ExecutePreprocessor.enabled=True Import.ipynb
    if [ $? != 0 ]; then
        echo "Importing $ID failed"
        exit -1
    fi
    mv Import.html $ID.html
done


for ID in M3{Id,Fou} M3_{01,02,03}; do
    export ID
    export BASEDIR="$HOME/Downloads/Phaselift_2017_8_4/3x3 phaselift"
    jupyter nbconvert --to html --ExecutePreprocessor.enabled=True Import.ipynb
    if [ $? != 0 ]; then
        echo "Importing $ID failed"
        exit -1
    fi
    mv Import.html $ID.html
done


for ID in M5{Id,Fou,Swap} M5_{01,02,03}; do
    export ID
    export BASEDIR="$HOME/Downloads/Phaselift_2017_8_4/5x5 phaselift"
    export BASEDIR_OFFSET="$HOME/Downloads/Phaselift_2017_14_4/5x5 phaselift"
    jupyter nbconvert --to html --ExecutePreprocessor.enabled=True Import.ipynb
    if [ $? != 0 ]; then
        echo "Importing $ID failed"
        exit -1
    fi
    mv Import.html $ID.html
done
