#!/usr/bin/env python3

"""
hyperpartisan text-to-process

hyperpartisan is a command line tool that evaluates an ensemble
model that is trained to estimate the "hyperpartisan" category
of a news document.
"""

# This tool requires an extensive Python environment,
# and various populated directories.
# It should be possible to create a suitable Python environment
# using conda and the conda.yaml file:
#   conda env create --file conda.yaml
# The conda.yaml file has a default prefix (name) of "hyperpartisan",
# if you want to use an alternate prefix, add a --prefix argument.

# This tool requires trained models in HDF5 files to be
# in the "prediction_models" directory.
# After you have trained models in the saved_models directory
# (see the README.md for how to train models):
# Populate the directory, on Unix with a GNU-ish sort:
#   cp $(ls saved_models/*.hdf5 | sort -r | sed 3q) prediction_models/

# This tool also requires a populated elmo directory.
# See the README.md


# https://docs.python.org/3.5/library/glob.html
import glob

# https://docs.python.org/3.5/library/json.html
import json

# https://docs.python.org/3.5/library/sys.html
import sys

# https://docs.python.org/3.5/library/xml.etree.elementtree.html
import xml.etree.ElementTree

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

# Avoid "Using TensorFlow backend" stderr stutter.
# Thanks, random GitHub person: https://github.com/keras-team/keras/issues/1406#issuecomment-466135951
import contextlib

with contextlib.redirect_stderr(open(os.devnull, "w")):
    import keras.preprocessing

import numpy
import spacy
import sklearn.preprocessing

from Preprocessing import nlp
from Preprocessing import utils
from Preprocessing import xml2line
from Preprocessing import line2elmo2

import ensemble_pred


def hyperpartisan(text):
    """
    Given the text of an article, as a Python string,
    return a probability that the article is hyperpartisan.
    As estimated by the models in the directory "prediction_models".
    """

    article = {"id": "command line text", "xml": text}

    add_article_sent(article)

    vectors = elmo_embedding(article)

    prediction = apply_ensemble_model(vectors)

    return prediction


def add_article_sent(article):
    """
    To the article represented by `article`, add
    a "article_sent" key which is the tokenised text
    of the article joined with spaces into a single string.
    (and perform other processing necessary to get to this point)
    """

    article["et"] = xml.etree.ElementTree.fromstring(
        '<article title="command line text" />'
    )

    pipeline = xml2line.create_pipeline()
    nerror = utils.run_pipeline(pipeline, article)
    article["nerror"] = nerror

    del article["et"]

    return article


def elmo_embedding(article):
    """
    Convert an article (using the text in its "article_sent" key)
    to embedded ELMo vectors.

    A 2-dimensional array is returned,
    with one row per sentence.
    """

    sents = article["article_sent"].split(" <splt> ")

    elmo = line2elmo2.create_elmo("original", False)
    vectors = line2elmo2.elmo_one_article(elmo, sents, 200, 200, batchsize=50)
    return vectors


def apply_ensemble_model(vectors):
    """
    Given a list of vectors representing an article,
    apply the ensemble model and return a score.
    """

    model = ensemble_pred.create_ensemble_from_files(
        sorted(glob.glob("prediction_models/*.hdf5"))
    )

    padded_vectors = keras.preprocessing.sequence.pad_sequences(
        [vectors], maxlen=200, dtype="float32"
    )[0]

    data = numpy.array([padded_vectors])

    predicts = model.predict(data)
    return predicts[0].tolist()[0]


def check_prerequisites():
    """
    Check for common problems with prequisites.
    Prints to stderr, any problems that it finds.

    Return True is everything that is check is okay;
    Return False otherwise (one or more checks failed).
    """

    reports = []

    # Check NLTK stopwords
    try:
        nlp.init_stopwords()
    except LookupError as err:
        reports.append(("The NLTK stopwords need to be downloaded, see README.md", err))

    # Check spaCy model
    model_name = "en_core_web_sm"
    try:
        spacy.load(model_name)
    except OSError as err:
        reports.append(("The spaCy model {!r} needs to be downloaded, see README.md".format(model_name), err))


    # Check ELMo model
    ELMo_model = "original"
    try:
        line2elmo2.create_elmo(ELMo_model, False)
    except FileNotFoundError as err:
        reports.append(("The ELMo model needs to be downloaded, see README.md", err))

    # Check the trained hyper-partisan models
    files = glob.glob("prediction_models/*.hdf5")
    if len(files) == 0:
        reports.append(("Trained model files need to be in the prediction_models directory, see hyperpartisan.py", ""))


    if not reports:
        return True

    for report in reports:
        friendly, exception = report
        print(exception, file=sys.stderr)
        print(friendly, file=sys.stderr)

    return False


def main(argv=None):
    if argv is None:
        argv = sys.argv

    arg = argv[1:]

    if not check_prerequisites():
        return 4

    text = " ".join(arg)
    score = hyperpartisan(text)
    print(json.dumps({"hyperpartisan_probability": score}))


if __name__ == "__main__":
    sys.exit(main())
