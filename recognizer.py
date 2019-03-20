from bottle import route, run, template, redirect, request, HTTPResponse
import simplejson as json
import sqlite3
import face_recognition


@route("/recognize", method='POST')
def somethingjson():
    upload = request.files.get('file', '')
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'
    upload.save("face.png")

    result = face_recognition.doIt()
    print(result)
    body = json.dumps({'name': result})
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


# テスト用のサーバをlocalhost:8080で起動する
run(host="localhost", port=8080, debug=True, reloader=True)
