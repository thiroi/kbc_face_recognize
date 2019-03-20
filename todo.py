from bottle import route, run, template, redirect, request, HTTPResponse
import simplejson as json
import sqlite3
import face_recognition


@route("/recognize")
def somethingjson():
    result = face_recognition.doIt()
    print(result)
    body = json.dumps({'name': result})
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


# テスト用のサーバをlocalhost:8080で起動する
run(host="localhost", port=8080, debug=True, reloader=True)
