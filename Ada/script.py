import os
import shutil

def result_dir():
    '''
    Delete the folder 删除result文件夹
    :return:  None
    '''
    if os.path.exists('result'):
        shutil.rmtree('result')


if __name__ == '__main__':
    result_dir()