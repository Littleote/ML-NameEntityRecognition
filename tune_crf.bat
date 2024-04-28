@echo off

:: default configuration parameters
if not defined BASEDIR (
    set BASEDIR=.
)
if not defined PYTHON (
    set PYTHON=python
)

:: change configuration
if exist config.bat (
    call config
) else (
    echo You can set your configuration at "config.bat"
    echo :: Your configuration>config.bat
    echo :: set BASEDIR=.>>config.bat
    echo :: set PYTHON=python>>config.bat
)

:: convert datasets to feature vectors
echo "Extracting features..."
%PYTHON% extract-features.py %BASEDIR%/data/train/ > train.feat
%PYTHON% extract-features.py %BASEDIR%/data/devel/ > devel.feat

del tune.csv

for %%f in (1 2 5 10 20 50) do (
    for %%c in (0.001 0.01 0.1 1 10 100) do (
        echo Testing feature.minfreq=%%f, c2=%%c
        :: train CRF model
        %PYTHON% train-crf.py model.crf feature.minfreq %%f c2 %%c max_iterations 200 < train.feat
        :: run CRF model
        %PYTHON% predict.py model.crf < devel.feat > devel.out
        :: evaluate CRF results
        %PYTHON% evaluator.py NER %BASEDIR%/data/devel devel.out > devel.stats

        %PYTHON% summarize.py devel.stats tune.csv feature.minfreq %%f c2 %%c
    )
)