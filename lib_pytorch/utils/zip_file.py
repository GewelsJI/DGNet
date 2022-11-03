import zipfile
import os.path
import os


class ZFile(object):
    # pip install zipfile36
    def zip_file(fs_name, fz_name):
        """
        从压缩文件
        :param fs_name: 源文件名
        :param fz_name: 压缩后文件名
        :return:
        """
        flag = False
        if fs_name and fz_name:
            try:
                with zipfile.ZipFile(fz_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(fs_name)
                    print(
                        "%s is running [%s] " %
                        (currentThread().getName(), fs_name))
                    print('压缩文件[{}]成功'.format(fs_name))
                if zipfile.is_zipfile(fz_name):
                    os.remove(fs_name)
                    print('删除文件[{}]成功'.format(fs_name))
                flag = True
            except Exception as e:
                print('压缩文件[{}]失败'.format(fs_name), str(e))

        else:
            print('文件名不能为空')
        return {'file_name': fs_name, 'flag': flag}

    def unzip_file(fz_name, path):
        """
        解压缩文件
        :param fz_name: zip文件
        :param path: 解压缩路径
        :return:
        """
        flag = False

        if zipfile.is_zipfile(fz_name):  # 检查是否为zip文件
            with zipfile.ZipFile(fz_name, 'r') as zipf:
                zipf.extractall(path)
                # for p in zipf.namelist():
                #     # 使用cp437对文件名进行解码还原， win下一般使用的是gbk编码
                #     p = p.encode('cp437').decode('gbk')  # 解决中文乱码
                #     print(fz_name, p,path)
                flag = True

        return {'file_name': fz_name, 'flag': flag}

# ZFile.unzip_file(fz_name='/home/users/u7248002/project/DGNet/dataset/TrainSet.zip', path='/home/users/u7248002/project/DGNet/dataset')
# ZFile.unzip_file(fz_name='/home/users/u7248002/project/DGNet/dataset/TestSet.zip', path='/home/users/u7248002/project/DGNet/dataset')
ZFile.unzip_file(fz_name='/home/users/u7248002/project/DGNet/lib_pytorch/snapshot/dgnet.zip', path='/home/users/u7248002/project/DGNet/lib_pytorch/snapshot/')