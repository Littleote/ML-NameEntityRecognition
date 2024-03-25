#! /usr/bin/python3

import sys
from os import listdir, path
from typing import Iterator
# import re

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize


## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets


def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag -----------
##  Find out whether given token is marked as part of an entity in the XML


def get_tag(token, spans):
    (form, start, end) = token
    for spanS, spanE, spanT in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT

    return "O"


## --------- Feature extractor -----------
## -- Extract features for each token in given sentence


def runWindow(
    tokens: list[list],
    index: int,
    window: list[int],
    *fun_args: list,
) -> list[str]:
    bos, eos = True, True
    features: list[str] = []
    for delta in window:
        k = index + delta
        ext = ""
        if delta < 0:
            ext = f"Prev{-delta}"
            if k < 0:
                if bos:
                    bos = False
                    features.append("BoS")
                continue
        if delta > 0:
            ext = f"Next{delta}"
            if k >= len(tokens):
                if eos:
                    eos = False
                    features.append("BoS")
                continue
        token = tokens[k][0]
        for fun, *args in fun_args:
            fun(features, token, *args, ext=ext)
    return features


def addForm(feat: list[str], token: str, *, ext: str = ""):
    feat.append(f"form{ext}={token}")


def addFormCasing(feat: list[str], token: str, *, ext: str = ""):
    feat.append(f"form{ext}={token.lower()}")
    if token.islower():
        feat.append(f"lower{ext}=True")
    elif token[1:].islower():
        feat.append(f"title{ext}=True")
    elif token.isupper():
        feat.append(f"upper{ext}=True")


def addSuffix(
    feat: list[str],
    token: str,
    size: int | Iterator[int],
    *,
    ext: str = "",
):
    if isinstance(size, int):
        size = [size]
    for s in size:
        feat.append(f"suf{s}{ext}={token[-s:].lower()}")


def addEndCharacters(feat: list[str], token: str, size: int, *, ext: str = ""):
    for s in range(1, min(size, len(token)) + 1):
        feat.append(f"last{s}{ext}={token[-s].lower()}")


def extract_features(tokens):
    # for each token, generate list of features and add it to the result
    result = []
    for k in range(0, len(tokens)):
        features = runWindow(
            tokens,
            k,
            [-1, 0, +1],
            # (addForm,),
            (addFormCasing,),
            (addSuffix, 3),
            # (addEndCharacters, 3),
        )
        result.append(features)

    return result


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]
IS_CRF = len(sys.argv) > 2 and sys.argv[2].upper() == "CRF"

# process each file in directory
for f in listdir(datadir):
    # parse XML file, obtaining a DOM tree
    tree = parse(path.join(datadir, f))

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        spans = []
        stext = s.attributes["text"].value  # get sentence text
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            typ = e.attributes["type"].value
            spans.append((int(start), int(end), typ))

        # convert the sentence to a list of tokens
        tokens = tokenize(stext)
        # extract sentence features
        features = extract_features(tokens)

        # print features in format expected by crfsuite trainer
        for i in range(0, len(tokens)):
            # see if the token is part of an entity
            tag = get_tag(tokens[i], spans)
            info = [
                sid,
                tokens[i][0],
                tokens[i][1],
                tokens[i][2],
                tag,
            ]
            if IS_CRF:
                info = [tag]
            print(
                *info,
                "\t".join(features[i]),
                sep="\t",
            )

        # blank line to separate sentences
        if not IS_CRF:
            print()
