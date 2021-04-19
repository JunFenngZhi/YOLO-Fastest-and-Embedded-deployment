from xml.dom.minidom import Document
import os
import cv2



# 此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    files = os.listdir(txtPath)
    for i, name in enumerate(files):  # name读出来的是所有文件(jpg和txt)
        if i % 100 == 0:
            print("当前处理：", name)
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签 (顶层元素，类似于根节点)
        xmlBuilder.appendChild(annotation)

        # 读取图片信息及其对应的标签
        txtFile = open(txtPath + name[0:-4] + ".txt")  # , encoding='utf-8',txtPath + name
        txtList = txtFile.readlines()
        txtFile.close()
        img = cv2.imread(picPath + name[0:-4] + ".jpg")
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("foramt")  # format标签
        foldercontent = xmlBuilder.createTextNode("VOC2007")  # 文件名
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # format标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList[3:]:  # 只处理目标信息的那几行
            oneline = j.strip().replace(':', ' ').replace('{', ' ').replace(',', ' ').replace('}', '')  # 去除标点符号
            oneline = oneline.split(' ')

            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(str(oneline[1]))
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束
            '''
            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束
            '''
            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签  (注意label坐标是【row,col】对应的是【y,x】)
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = oneline[3]  # int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = oneline[2]  # int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = oneline[5]  # int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = oneline[4]  # int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束
            annotation.appendChild(object)  # object标签结束

        # 改名字
        old_img_name = os.path.join(picPath, name[0:-4] + ".jpg")
        new_img_name = os.path.join(picPath, "Cloud_2m_" + name[0:-4] + ".jpg")
        os.rename(old_img_name, new_img_name)
        old_txt_name = os.path.join(txtPath, name[0:-4] + ".txt")
        new_txt_name = os.path.join(txtPath, "Cloud_2m_" + name[0:-4] + ".txt")
        os.rename(old_txt_name, new_txt_name)

        f = open(xmlPath + "Cloud_2m_" + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


makexml(txtPath="E:\Graduate_Design\Data\\Cloud\\2m\\txt\\",
        xmlPath="E:\Graduate_Design\Data\\Cloud\\2m\\xml\\",
        picPath="E:\Graduate_Design\Data\\Cloud\\2m\\img\\")  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径

'''    
舰船数据集中，类别对应id
dict = {"carrier": '0', "defender": '1', "destroyer": '2'}

name[0:-4] 针对的是原始名字0000.jpg。假如名字修改了，则应该更改
'''