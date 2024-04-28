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
%PYTHON% extract-features.py %BASEDIR%/data/train/ CRF > train.clf.feat
%PYTHON% extract-features.py %BASEDIR%/data/devel/ > devel.feat

:: train CRF model
echo "Training CRF model..."
%PYTHON% train-crf.py model.crf < train.feat
:: run CRF model
echo "Running CRF model..."
%PYTHON% predict.py model.crf < devel.feat > devel-CRF.out
:: evaluate CRF results
echo "Evaluating CRF results..."
%PYTHON% evaluator.py NER %BASEDIR%/data/devel devel-CRF.out CM-CRF.png > devel-CRF.stats

:: train Scikit-learn model
echo "Training Scikit-learn model..."
%PYTHON% train-sklearn.py model.joblib vectorizer.joblib < train.clf.feat
:: run Scikit-learn model
echo "Running Scikit-learn model..."
%PYTHON% predict-sklearn.py model.joblib vectorizer.joblib < devel.feat > devel-NB.out
:: evaluate Scikit-learn results 
echo "Evaluating Scikit-learn results..."
%PYTHON% evaluator.py NER %BASEDIR%/data/devel devel-NB.out CM-NB.png > devel-NB.stats

:: remove auxiliary files.
del train.clf.feat

