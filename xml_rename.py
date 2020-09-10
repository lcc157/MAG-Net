import xml.dom.minidom
import os
import xml.etree.ElementTree
path='D:\\1' # xml文件存放路径
sv_path='D:\\11' # 修改后的xml文件存放路径
files=os.listdir(path)
for xmlFile in files:
	xmlname = os.path.splitext(xmlFile)[0]
	print(xmlname)
	dom=xml.dom.minidom.parse(os.path.join(path,xmlFile)) #打开xml文件，送到dom解析
	root=dom.documentElement #得到文档元素对象
	#item=root.getElementsByTagName('path') #获取path这一node名字及相关属性值
	#a,b=os.path.splitext(xmlFile) #分离出文件名a
	#for i in item:
	     #i.firstChild.data=a + '.jpg'
	#with open(os.path.join(sv_path,xmlFile),'w') as fh:
		#dom.writexml(fh)
	root.getElementsByTagName('filename')[0].firstChild.data = xmlname + '.jpg'
    #specific = sv_path + xmlFile
	with open(os.path.join(sv_path, xmlFile), 'w') as fh:
	      dom.writexml(fh)