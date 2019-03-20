from bottle import route, run, template, redirect, request, HTTPResponse
import simplejson as json
import sqlite3
import face_recognition

# データベースに接続
dbname = "todo.db"
conn = sqlite3.connect(dbname)
c = conn.cursor()

try:
    # テーブルの作成
    c.execute("DROP TABLE IF EXISTS todo_list")
    c.execute(
        "CREATE TABLE IF NOT EXISTS todo_list (id INTEGER PRIMARY KEY, todo TEXT)")
    # プログラムから試しに１つだけToDoを追加しておく
    c.execute("INSERT INTO todo_list VALUES (1, 'あらかじめ挿入するTODO')")
except sqlite3.Error as e:
    print('sqlite3.Error occurred:', e.args[0])

conn.commit()
conn.close()

# / にアクセスしたら、index関数が呼ばれる
@route("/")
def index():
    todo_list = get_todo_list()
    return template("index", todo_list=todo_list)


@route("/jsontest")
def somethingjson():
    result = face_recognition.doIt()
    print(result)
    body = json.dumps({'message': result})
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r

# methodにPOSTを指定して、add関数を実装する
@route("/add", method="POST")
def add():
    todo = request.forms.getunicode("todo_list")
    save_todo(todo)
    return redirect("/")

# @routeデコレータの引数で<xxxx>と書いた部分は引数として関数に引き渡すことができます。
# intは数字のみ受け付けるフィルタ
@route("/delete/<todo_id:int>")
def delete(todo_id):
    delete_todo(todo_id)
    return redirect("/")


def get_todo_list():
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    select = "select id, todo from todo_list"
    c.execute(select)
    todo_list = []
    for row in c.fetchall():
        todo_list.append({
            "id": row[0],
            "todo": row[1]
        })
    conn.close()
    return todo_list


def save_todo(todo):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    insert = "insert into todo_list(todo) values(?)"
    c.execute(insert, (todo,))  # todoのあとにカンマをつけないとなぜかエラーになる
    conn.commit()
    conn.close()


def delete_todo(todo_id):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    delete = "delete from todo_list where id=?"
    c.execute(delete, (todo_id,))
    conn.commit()
    conn.close()


# テスト用のサーバをlocalhost:8080で起動する
run(host="localhost", port=8080, debug=True, reloader=True)
