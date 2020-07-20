import sqlite3

def connecter():
	#连接数据库
	conn = sqlite3.connect('DRWord.db')
	cursor = conn.cursor();
	return conn, cursor

def dbclose(conn, cursor):
	#关闭数据库连接
	cursor.close()
	conn.close()

#table user
def createtable(tablename):
	#创建数据表
	conn, cur = connecter()
	sql_create = 'create table ' + tablename +' (id int primary key not null, username text, password text)'
	cur.execute(sql_create)
	conn.commit()
	dbclose(conn,cur)

def insertuser(username, password, tablename='user'):
	'''
	server_run.register_check
	创建新用户
	'''
	conn,cur = connecter()
	sql_select_exit = 'select * from ' + tablename +' where username=?'
	cur.execute(sql_select_exit, (username,))
	if len(cur.fetchall()) != 0 :
		dbclose(conn,cur)
		return 'user already exit!'
	else: 
		sql_select = 'select * from ' + tablename
		cur.execute(sql_select)
		idnum = len(cur.fetchall()) + 1
		sql_insert = 'insert into user values(?,?,?)'
		cur.execute(sql_insert, (idnum, username, password))
		conn.commit()
		dbclose(conn,cur)
		return 'OK'

def selectuser(username, password):
	'''
	server_run.login_check
	确定用户是否合法
	'''
	conn, cur = connecter()
	sql_select = 'select * from user where username=? and password=?'
	cur.execute(sql_select, (username,password))
	values = cur.fetchall()
	if len(values) != 0:
		print(values[0])
		dbclose(conn,cur)
		return 'OK'
	else:
		#values = cur.fetchall()
		#print(len(values))
		dbclose(conn,cur)		
		return 'username or password wrong!'

def updatepassword(username, password, newpassword, tablename='user'):
	'''
	更新用户信息
	'''
	conn, cur = connecter()
	sql_select = 'select * from ' + tablename + ' where username=?'
	cur.execute(sql_select, (username,))
	values = cur.fetchall()

	if len(values) == 0:
		dbclose(conn,cur)
		return 'user does not exit!'
	elif values[0][2] != password:
		dbclose(conn,cur)
		return 'old password is wrong!'
	else:
		sql_update = 'update user set password=? where username=?'
		cur.execute(sql_update, (newpassword, username))
		conn.commit()
		dbclose(conn, cur)
		return 'OK'

def getuser(username):
	'''
	获取指定用户的信息
	'''
	conn, cur = connecter()
	sql_select = "select * from user where username=?"
	cur.execute(sql_select, (username,))
	values = cur.fetchall()
	if len(values) != 0:
		dbclose(conn, cur)
		return values
	else:
		return 'user does not exit!'

def deleteuser(username):
	'''
	删除指定用户
	'''
	conn, cur = connecter()
	sql_delete = "delete from user where username=?"
	cur.execute(sql_delete, (username,))
	conn.commit()
	dbclose(conn, cur)
	return 'OK'

def checkusertable():
	'''
	查看user表所有数据
	'''
	conn, cur = connecter()
	sql_select = "select * from user"
	cur.execute(sql_select)
	values = cur.fetchall()
	if len(values) != 0:
		dbclose(conn, cur)
		return values
	else:
		return 'user does not exit!'




#table userinfo
def insertuserinfo(username, userhead="initheadpic.jpg"):
	'''
	创建一个新的用户
	'''
	conn, cur = connecter()
	sql_insert ="insert into userinfo (username, userhead) values(?,?)"
	cur.execute(sql_insert, (username, userhead))
	conn.commit()
	dbclose(conn, cur)
	return 'OK'

def updateuserinfo(username, realname=None, phonenumber=None, email=None, sex=None, userhead="initheadpic.jpg"):
	'''
	更新用户信息
	'''
	conn, cur = connecter()
	sql_update = "update userinfo set realname=?, phonenumber=?, email=?, sex=?, userhead=? where username=?"
	cur.execute(sql_update, (realname, phonenumber, email, sex, userhead, username))
	conn.commit()
	dbclose(conn, cur)
	return 'OK'

def getuserinfo(username):
	'''
	查看指定用户的信息
	'''
	conn, cur = connecter()
	sql_select = "select * from userinfo where username=?"
	cur.execute(sql_select, (username,))
	values = cur.fetchall()
	if len(values) != 0:
		dbclose(conn, cur)
		return values
	else:
		return 'user does not exit!'


def deleteuserinfo(username):
	'''
	删除指定用户
	'''
	conn, cur = connecter()
	sql_delete = "delete from userinfo where username=?"
	cur.execute(sql_delete, (username,))
	conn.commit()
	dbclose(conn, cur)
	return 'OK'

def checkuserinfotable():
	'''
	查看userinfo表中的所有数据
	'''
	conn, cur = connecter()
	sql_select = "select * from userinfo"
	cur.execute(sql_select)
	values = cur.fetchall()
	if len(values) != 0:
		dbclose(conn, cur)
		return values
	else:
		return 'user does not exit!'

if __name__ == '__main__':
	#user表相关操作
	#createtable('user')
	#print(insertuser('wsw', 'wsw123', 'user'))
	#deleteuser('test')
	print(checkusertable())

	
	#userinfo表相关操作
	#print(updatepassword('wsw','wsw','wsw123'))
	#print(selectuser('wsw', 'wsw123'))
	#print(insertuserinfo('admin'))
	#print(insertuserinfo('ruby'))
	#print(updateuserinfo('wsw'))
	#print(getuserinfo('wsw'))
	#deleteuserinfo('test')
	print(checkuserinfotable())