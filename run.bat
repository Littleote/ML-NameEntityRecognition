@echo off

set BASEDIR=.
set PYTHON="C:\Users\Zhengyong Ji\anaconda3\python.exe"

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
%PYTHON% evaluator.py NER %BASEDIR%/data/devel devel-CRF.out > devel-CRF.stats

:: train Naive Bayes model
echo "Training Naive Bayes model..."
%PYTHON% train-sklearn.py model.joblib vectorizer.joblib < train.clf.feat
:: run Naive Bayes model
echo "Running Naive Bayes model..."
%PYTHON% predict-sklearn.py model.joblib vectorizer.joblib < devel.feat > devel-NB.out
:: evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
%PYTHON% evaluator.py NER %BASEDIR%/data/devel devel-NB.out > devel-NB.stats

:: remove auxiliary files.
del train.clf.feat

