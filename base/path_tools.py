"""为整个工程项目提供统一的绝对路径
todo 大白话：这个文件就是"导航仪"——不管你从哪个文件夹运行代码，它都能帮你找到项目根目录
todo 因为不同人电脑不同，项目放的位置也不同，用绝对路径才能保证代码到处都能跑
"""

import os
# todo os 是 Python 自带的操作系统工具包，这里用来操作文件路径

def get_project_root():
    # todo 这个函数就是把所有的文件路径都变成项目的绝对路径D:\RAG_EDU_FQA
    """获取工程根目录
    无论脚本在哪个目录运行，都能返回正确的根目录
    原理：基于当前文件的绝对路径，向上推导到工程根目录
    
    todo 大白话：
    todo 这个文件在 base/path_tools.py，那项目根目录就是往上一层（base的父目录）
    todo 工作原理：拿到本文件的路径 → 去掉文件名得到 base/ → 再去掉一层得到项目根目录
    """
    # 当前文件（path_utils.py）的绝对路径
    # todo __file__ 是Python自带的变量，表示当前这个文件的路径，abspath把它变成完整绝对路径
    current_file = os.path.abspath(__file__)
    # print(current_file)
    # 当前文件所在目录（base/）
    # todo dirname 把路径的最后一部分（文件名）砍掉，只留目录部分
    current_dir = os.path.dirname(current_file)
    # print(current_dir)
    project_root = os.path.dirname(current_dir) # todo 再砍掉一层 base/，就得到项目根目录
    # print(project_root)
    return project_root


def get_abs_path(relative_path):
    """
    将工程内的相对路径转为绝对路径（统一路径基准）
    :param relative_path: 相对于工程根目录的路径，如 "base/config.py"
    :return: 绝对路径
    
    todo 大白话：你把"从项目根目录出发的路径"给我，我把它拼成"从C盘/D盘开始的完整路径"
    todo 比如传 "rag_qa/data/file.pdf" → 返回 "D:/RAG_EDU_FQA/rag_qa/data/file.pdf"
    """
    project_root = get_project_root() # todo 先找到项目根目录在哪
    # print(project_root)
    # 拼接绝对路径
    # todo os.path.join 会把两段路径用正确的斜杠拼起来（Windows用\，Linux用/）
    abs_path = os.path.join(project_root, relative_path)
    # print(abs_path)
    return abs_path



if __name__ == '__main__':
    get_project_root()
    get_abs_path(relative_path='main.py')