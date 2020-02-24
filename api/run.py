import sys
from app import CreateApp
from time import sleep
from flask import Flask, render_template
import logging

app = CreateApp('config')

if __name__ == '__main__':
	app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])