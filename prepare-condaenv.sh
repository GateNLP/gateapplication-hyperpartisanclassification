#!/bin/sh

# A POSIX shell script to prepare the condaenv directory

set -e -u

# For conda
# conda env create --prefix condaenv --file conda.yaml
# For NLTK stopwords
condaenv/bin/python -m nltk.downloader -d ./condaenv/nltk_data stopwords
