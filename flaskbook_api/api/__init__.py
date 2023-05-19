from flask import Blueprint, jsonify, request

from flaskbook_api.api import calculation

api = Blueprint("api", __name__)