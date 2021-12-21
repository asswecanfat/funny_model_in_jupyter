from pywebio import *

def t():

    name = input.input("what's your name")
    output.put_text("hello", name)

start_server(t)