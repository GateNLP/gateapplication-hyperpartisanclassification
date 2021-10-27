# hyperpartisan_elg.py
#
# This script defines a simple Flask application that follows the internal API
# specifications of the European Language Grid (ELG):
#
# https://european-language-grid.readthedocs.io/en/release1.1.2/all/A3_API/LTInternalAPI.html
#
# In addition to the regular dependencies of the hyperpartisan classifier
# itself, you must also install flask and Flask-JSON
#
#     pip install flask Flask-JSON
#
# The app presents an endpoint at /process which accepts requests in the ELG
# "text" request format:
#
#     {
#       "type":"text",
#       "content":"<Text to be analysed>"
#     }
#
# runs the hyperpartisan classifier on the "content" text and returns a
# response in the "annotations" format
#
#     {
#       "response":{
#         "type":"annotations",
#         "annotations":{},
#         "features":{
#           "hyperpartisan_probability":number-between-0-and-1
#         }
#       }
#     }
#
# Errors are returned in the standard ELG "failure" message format.
#

from flask import Flask, request
from flask_json import FlaskJSON, JsonError, json_response, as_json
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
import cgi
import os
import json
import sys
import traceback

import hyperpartisan

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSON_ADD_STATUS"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv('REQUEST_SIZE_LIMIT', 50000))

json_app = FlaskJSON(app)

@json_app.invalid_json_error
def invalid_request_error(e):
    """Generates a valid ELG "failure" response if the request cannot be parsed"""
    return {'failure':{ 'errors': [
        { 'code':'elg.request.invalid', 'text':'Invalid request message' }
    ] } }, 400

app.register_error_handler(BadRequest, invalid_request_error)
app.register_error_handler(UnicodeError, invalid_request_error)

@app.errorhandler(RequestEntityTooLarge)
def request_too_large(e):
    """Generates a valid ELG "failure" response if the request is too large"""
    return {'failure':{ 'errors': [
        { 'code':'elg.request.too.large', 'text':'Request size too large' }
    ] } }, 400


@app.route('/process', methods=['POST'])
@as_json
def process_request():
    """Main request processing logic - accepts a JSON request and returns a JSON response."""
    ctype, type_params = cgi.parse_header(request.content_type)
    if ctype == 'application/json':
        data = request.get_json()
        # sanity checks on the request message
        if data.get('type') != 'text' or 'content' not in data:
            raise BadRequest()
        content = data['content']
    else:
        raise BadRequest()

    try:
        score = hyperpartisan.hyperpartisan(content)
        return dict(response = {
            'type':'annotations',
            'features':dict(hyperpartisan_probability = score),
            'annotations':dict()
        })
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        # Convert any medcat exception into an ELG internal error
        raise JsonError(status_=500, failure={ 'errors': [
            { 'code':'elg.service.internalError', 'text':'Internal error during processing: {0}',
              'params':[traceback.format_exception_only(exc_type, exc_value)[-1]] }
        ]})


if __name__ == '__main__':
    app.run()
