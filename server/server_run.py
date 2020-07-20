from flask import Flask, render_template, request, session, redirect, make_response, url_for, json,jsonify
import json
import os
from flask_cors import CORS
import cv2
#多进程
from multiprocessing import Process, Manager

#添加系统路径
import sys
sys.path.append('/home/wsw/workplace/deeplearning/DRWord/algorithm/')

#调用文本识别算法
#from DRWord_en.predict_en import predicten
#from DRWord_ch.predict_mtwi import predictmtwi
import toolbar
import database


app = Flask(__name__)
#设置cookie密钥
app.secret_key = '@#$DS$%^GT^'
CORS(app, resources=r'/*')
#设置json返回的文字格式
app.config['JSON_AS_ASCII'] = False

'''
#current_user_list = []
from werkzeug.serving import run_with_reloader
def restart():
	app = Flask()
'''

@app.route('/login_check', methods=['GET', 'POST'])  # 路由
def login_check():
	'''
	检查登录信息是否合法
	'''
	if request.method == 'POST':
		#获取网页端传输的数据
		a = request.get_data()
		data = json.loads(a)
		'''
		if data['u'] in current_user_list:
			return 'user already login!'
		'''
		#查看数据库中用户名与密码是否匹配
		result = database.selectuser(data['u'], data['p'])
		if result == 'OK':
			#current_user_list.append(data['u'])
			#验证成功后返回数据到网页端，并设置cookie信息
			response = make_response('OK')
			response.set_cookie('username', data['u'])
			return response
		else:
			return json.dumps(result)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
	'''
	退出当前登录
	'''
	#获取当前用户名
	username = request.cookies.get("username")
	#current_user_list.remove(username)
	#退出后删除cookie信息并返回到登录页
	response = make_response(redirect(url_for('login')))
	response.delete_cookie('username')
	return response

@app.route('/closeaccount')
def closeaccount():
	'''
	删除当前账号
	'''
	username = request.cookies.get("username")

	database.deleteuser(username)
	database.deleteuserinfo(username)
	toolbar.removedir(username)
	
	response = make_response(redirect(url_for('login')))
	response.delete_cookie('username')
	return response

@app.route('/register_check', methods=['GET', 'POST'])
def register_check():
	'''
	检查注册信息是否合法
	'''
	if request.method == 'POST':
		#获取网页端传输的信息
		a = request.get_data()
		data = json.loads(a)

		print("register check : ", data)

		if(data['p']!=data['c']):
			result = 'password and checkword are not equal!'
		else:
			#在数据库中新建用户
			result = database.insertuser(data['u'], data['p'])
		if result == 'OK':
			#向userinfo表添加新用户并为新用户创建文件夹
			database.insertuserinfo(data['u'])
			toolbar.makeuserdir(data['u'])
			return 'OK'
		else:
			return json.dumps(result)


#handle the userinfo table
@app.route('/get_user_head', methods=['GET', 'POST'])
def get_user_head():
	'''
	从userinfo表中获取用户头像信息，用于在网页头部导航栏显示头像图片
	'''
	#获取用户名并从userinfo表中获取用户信息
	username = request.cookies.get('username')
	values = database.getuserinfo(username)
	
	#用字典保存数据，返回网页端时转换成json格式
	result = dict()
	result['username'] = values[0][0]
	result['userhead_pic'] = values[0][5]
	#print(result)

	return jsonify(result)

@app.route('/get_user_info', methods=['GET', 'POST'])
def get_user_info():
	'''
	获用户信息，用于在个人信息页面显示详细信息
	'''
	#获取用户名并从userinfo表中获取用户信息
	username = request.cookies.get('username')
	values = database.getuserinfo(username)

	#用字典保存数据，返回网页端时转换成json格式
	result = dict()
	result['username'] = values[0][0]
	result['realname'] = values[0][1]
	result['phonenumber'] = values[0][2]
	result['email'] = values[0][3]
	result['sex'] = values[0][4]
	result['userhead_image'] = values[0][5]
	#print(result)

	return jsonify(result)

@app.route('/update_user_info', methods=['GET', 'POST'])
def update_user_info():
	'''
	更新用户信息
	'''
	username = request.cookies.get('username')

	if request.method == 'POST':

		#获得网页端传输的图片
		userhead_pic = request.files['userhead_image_change']
		#创建用户头像图片名称
		userhead_pic_path = username + "_" + userhead_pic.filename[:-4] + "_headpic.jpg"
		#保存图片
		userhead_pic.save('/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + userhead_pic_path)

		#获取其他用户信息
		realname = request.form['realname']
		phonenumber = request.form['phonenumber']
		email = request.form['email']
		sex = request.form['sex']

		#在userinfo表中更新用户个人信息
		result = database.updateuserinfo(username, realname, phonenumber, email, sex, username+'/'+userhead_pic_path)  

		if result == 'OK':
			return 'OK'
		else:
			return 'something wrong!'

@app.route('/update_password', methods=['GET', 'POST'])
def update_password():
	'''
	更新用户密码
	'''
	if request.method == 'POST':
		a = request.get_data()
		data = json.loads(a)
		result = database.updatepassword(data['username'], data['oldpassword'], data['newpassword'])

		return result


#文本检测模块
@app.route('/detect_pic_en', methods=['GET', 'POST'])
def detect_pic_en():
	'''
	实现英文快速识别模块
	'''

	from DRWord_en.predict_en import predicten

	if request.method == 'POST':
		#获取网页端提供的图片
		img = request.files['dr_pic_choose']

		#print(request.form['testinfo'])
		#im = cv2.imread(img)
		username = request.cookies.get("username")
		img_path = '/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + img.filename
		img.save(img_path)

		#im_result, txt_result = predicten(img_path)
		#使用多线程，用于解决tensorflow不能自动释放显存资源
		manager = Manager()
		return_dic = manager.dict()
		p_en = Process(target=predicten, args=(img_path, return_dic))
		p_en.start()
		p_en.join()

		#获得检测结果
		im_result = return_dic['im_result_en']
		txt_result= return_dic['txt_result_en']

		#保存识别后的图片
		img_predict_name = img.filename[:-4] + '_icdar_predict.jpg'
		img_predict_path = '/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + img_predict_name
		cv2.imwrite(img_predict_path, im_result)

		print(txt_result)

		#返回结果
		result = dict()
		result['code'] = 'OK'
		result['dr_pic'] = img_predict_name
		result['dr_pic_result'] = txt_result

		return jsonify(result)

@app.route('/detect_pic_mtwi', methods=['GET', 'POST'])
def detect_pic_mtwi():
	'''
	实现网络图片识别模块
	'''

	from DRWord_ch.predict_mtwi import predictmtwi

	if request.method == 'POST':
		#获取网页端提供的图片
		img = request.files['dr_pic_choose']
		#im = cv2.imread(img)
		username = request.cookies.get("username")
		img_path = '/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + img.filename
		img.save(img_path)

		#im_result, txt_result = predictmtwi(img_path)
		#使用多线程，用于解决tensorflow不能自动释放显存资源
		manager = Manager()
		return_dic = manager.dict()
		p_mtwi = Process(target=predictmtwi, args=(img_path, return_dic))
		p_mtwi.start()
		p_mtwi.join()

		#获得检测结果
		im_result = return_dic['im_result_mtwi']
		txt_result= return_dic['txt_result_mtwi']

		#保存识别后的图片
		img_predict_name = img.filename[:-4] + '_mtwi_predict.jpg'
		img_predict_path = '/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + img_predict_name
		cv2.imwrite(img_predict_path, im_result)

		print(txt_result)

		#返回结果
		result = dict()
		result['code'] = 'OK'
		result['dr_pic'] = img_predict_name
		result['dr_pic_result'] = txt_result

		return jsonify(result)

@app.route('/detect_pic_rctw', methods=['GET', 'POST'])
def detect_pic_rctw():
	'''
	实现街景图片识别模块
	'''

	from DRWord_ch.predict_rctw import predictrctw

	if request.method == 'POST':
		#获取网页端提供的图片
		img = request.files['dr_pic_choose']
		#im = cv2.imread(img)
		username = request.cookies.get("username")
		img_path = '/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + img.filename
		img.save(img_path)

		#im_result, txt_result = predictrctw(img_path)
		#使用多线程，用于解决tensorflow不能自动释放显存资源
		manager = Manager()
		return_dic = manager.dict()
		p_rctw = Process(target=predictrctw, args=(img_path, return_dic))
		p_rctw.start()
		p_rctw.join()

		#获得检测结果
		im_result = return_dic['im_result_rctw']
		txt_result= return_dic['txt_result_rctw']

		#保存识别后的图片
		img_predict_name = img.filename[:-4] + '_rctw_predict.jpg'
		img_predict_path = '/home/wsw/workplace/deeplearning/DRWord/server/static/userpic/' + username + '/' + img_predict_name
		cv2.imwrite(img_predict_path, im_result)

		print(txt_result)

		#返回结果
		result = dict()
		result['code'] = 'OK'
		result['dr_pic'] = img_predict_name
		result['dr_pic_result'] = txt_result

		return jsonify(result)


#登录页面
@app.route('/login')
def login():
	return render_template('login.html')

#注册页面
@app.route('/register')
def register():
	return render_template('register.html')

#首页
@app.route('/index')
def index():
	#username = request.cookies.get("username")
	#print(username)
	return render_template('index.html')



#英文快速识别页面
@app.route('/drworden')
def drworden():
	#username = request.cookies.get("username")
	#print(username)
	return render_template('drworden.html')

#网络图片识别页面
@app.route('/drwordchmtwi')
def drwordchmtwi():
	#username = request.cookies.get("username")
	#print(username)
	return render_template('drwordchmtwi.html')

#网络图片识别页面
@app.route('/drwordchrctw')
def drwordchrctw():
	#username = request.cookies.get("username")
	#print(username)
	return render_template('drwordchrctw.html')

#个人信息页面
@app.route('/userinfo')
def userinfo():
	return render_template('userinfo.html')

#修改密码页面
@app.route('/changepassword')
def changepassword():
	return render_template('changepassword.html')

if __name__=="__main__":
	#启动服务器，允许远程访问 
	app.run(host='0.0.0.0', debug=True, threaded=True)
