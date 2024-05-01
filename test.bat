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
%PYTHON% extract-features.py %BASEDIR%/data/test/ > test.feat

:: train CRF model
echo "Training CRF model..."
%PYTHON% train-crf.py model.crf < train.feat
:: run CRF model
echo "Running CRF model..."
%PYTHON% predict.py model.crf < train.feat > train-CRF.out
%PYTHON% predict.py model.crf < devel.feat > devel-CRF.out
%PYTHON% predict.py model.crf < test.feat > test-CRF.out
:: evaluate CRF results
echo "Evaluating CRF results..."
%PYTHON% evaluator.py NER %BASEDIR%/data/train train-CRF.out CM-train-CRF.png > train-CRF.stats
%PYTHON% evaluator.py NER %BASEDIR%/data/devel devel-CRF.out CM-devel-CRF.png > devel-CRF.stats
%PYTHON% evaluator.py NER %BASEDIR%/data/test test-CRF.out CM-test-CRF.png > test-CRF.stats

:: train Scikit-learn model
echo "Training Scikit-learn model..."
%PYTHON% train-sklearn.py model.joblib vectorizer.joblib < train.clf.feat
:: run Scikit-learn model
echo "Running Scikit-learn model..."
%PYTHON% predict-sklearn.py model.joblib vectorizer.joblib < train.feat > train-NB.out
%PYTHON% predict-sklearn.py model.joblib vectorizer.joblib < devel.feat > devel-NB.out
%PYTHON% predict-sklearn.py model.joblib vectorizer.joblib < test.feat > test-NB.out
:: evaluate Scikit-learn results 
echo "Evaluating Scikit-learn results..."
%PYTHON% evaluator.py NER %BASEDIR%/data/train train-NB.out CM-train-NB.png > train-NB.stats
%PYTHON% evaluator.py NER %BASEDIR%/data/devel devel-NB.out CM-devel-NB.png > devel-NB.stats
%PYTHON% evaluator.py NER %BASEDIR%/data/test test-NB.out CM-test-NB.png > test-NB.stats

:: remove auxiliary files.
del train.clf.feat

