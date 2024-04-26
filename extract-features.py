#! /usr/bin/python3

import re
import sys
from os import listdir, path
from typing import Iterator
from xml.dom.minidom import parse

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

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
    mark_endings: bool = True,
    ext: str = "",
) -> list[str]:
    bos, eos = mark_endings, mark_endings
    features: list[str] = []
    for delta in window:
        k = index + delta
        _ext = ext
        if delta < 0:
            _ext += f"Prev{-delta}"
            if k < 0:
                if bos:
                    bos = False
                    features.append("BoS")
                continue
        if delta > 0:
            _ext += f"Next{delta}"
            if k >= len(tokens):
                if eos:
                    eos = False
                    features.append("EoS")
                continue
        token = tokens[k][0]
        for fun, *args in fun_args:
            fun(features, token, *args, ext=_ext)
    return features


def addWord(feat: list[str], token: str, lower: bool = True, *, ext: str = ""):
    if lower:
        feat.append(f"form{ext}={token.lower()}")
    else:
        feat.append(f"form{ext}={token}")


def addLength(feat: list[str], token: str, *, ext: str = ""):
    feat.append(f"length{ext}={len(token)}")


def addCasing(feat: list[str], token: str, *, ext: str = ""):
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


def addPrefix(
    feat: list[str],
    token: str,
    size: int | Iterator[int],
    *,
    ext: str = "",
):
    if isinstance(size, int):
        size = [size]
    for s in size:
        feat.append(f"pref{s}{ext}={token[:s].lower()}")


def addEndCharacters(feat: list[str], token: str, size: int, *, ext: str = ""):
    for s in range(1, min(size, len(token)) + 1):
        feat.append(f"last{s}{ext}={token[-s].lower()}")


def addStartCharacters(feat: list[str], token: str, size: int, *, ext: str = ""):
    for s in range(1, min(size, len(token)) + 1):
        feat.append(f"first{s}{ext}={token[s].lower()}")


def addNGram(feat: list[str], token: str, size: int, *, ext: str = ""):
    token = token.lower()
    # Remove duplicates
    ngrams = {token[i : i + size] for i in range(len(token) - (size - 1))}
    for ngram in ngrams:
        feat.append(f"{size}gram{ngram}{ext}=True")


MAPPINGS = {
    "vowels": r"(?i)(a|e|i|o|u)|([a-z])|([0-9])|(-)|.",
    "long": r"([a-z])|([A-Z])|([0-9])|(-)|.",
    "short": r"([a-z]+)|([A-Z]+)|([0-9]+)|(-+)|[^a-zA-Z0-9\-]+",
}


def mapping_sub(matchobj: re.Match) -> str:
    for i, group in enumerate(matchobj.groups(), 1):
        if group is not None:
            return chr(97 + i)
    return chr(97)


def addMapping(feat: list[str], token: str, mapping: str, *, ext: str = ""):
    feat.append(
        f"{mapping}Mapping{ext}={re.sub(MAPPINGS[mapping], mapping_sub, token)}"
    )


ALL_DRUGS: set[str] = set()
POSITIONAL_DRUGS: list[set[str]] = list()


def loadDictionary():
    global ALL_DRUGS, POSITIONAL_DRUGS
    drugs = pd.read_csv("drugbank/drugbank tokens.csv")
    POSITIONAL_DRUGS = list(range(1 + drugs["Token number"].max()))
    for position, drugs in drugs.groupby("Token number"):
        POSITIONAL_DRUGS[position] = set(drugs["Common name"].str.lower())
    ALL_DRUGS = set.union(*POSITIONAL_DRUGS)


def addDictionary(feat: list[str], token: str, *, ext: str = ""):
    token = token.lower()
    if token in ALL_DRUGS:
        for position, drugs in enumerate(POSITIONAL_DRUGS):
            if token in drugs:
                feat.append(f"Drug{position}{ext}=True")
    else:
        feat.append(f"NoDrug{ext}=True")


LEMMATIZER = WordNetLemmatizer()


def addLemma(feat: list[str], token: str, *, ext: str = ""):
    feat.append(f"lemma={LEMMATIZER.lemmatize(token)}")


def extract_features(tokens):
    # for each token, generate list of features and add it to the result
    ## To use the tags uncomment the following line
    # tags = [[t[1]] for t in pos_tag([token[0] for token in tokens])]
    result = []
    for k in range(0, len(tokens)):
        features = (
            ## #2 Best feature extraction
            # runWindow(
            #     tokens,
            #     k,
            #     [-1, 0, +1],
            #     (addWord,),
            #     (addLemma,),
            #     (addLength,),
            #     (addMapping, "short"),
            #     (addDictionary,),
            #     (addNGram, 2),
            # )
            ## #1 Best feature extraction
            runWindow(
                tokens,
                k,
                [-2, -1, 0, +1, +2],
                (addWord,),
                (addLemma,),
                (addLength,),
                (addMapping, "short"),
                (addDictionary,),
                (addSuffix, 3),
            )
            ## To use the tags uncomment the following line
            # + runWindow(
            #     tags,
            #     k,
            #     [-1, 0, +1],
            #     (addWord, False),
            #     mark_endings=False,
            #     ext="Tag",
            # )
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

loadDictionary()
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
