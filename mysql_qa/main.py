
# todo MySQL问答系统入口模块：通过 BM25 + Softmax 算法在 MySQL 数据库里搜索和用户问题最相似的已有问答对
# todo 大白话：用户提一个问题，系统去题库里找最像的问题，把对应的答案返回给用户
# todo BM25：一种经典的文本检索算法，用来计算两个文本之间的相似度得分
# todo Softmax：一种数学函数，把一组原始得分转换成概率值，让结果更容易理解和比较
from mysql_qa.core_search import Bm25SoftmaxSearch
from mysql_qa.utils.mysql_client import database_init
from base import logger
import time


# todo MySQLQaSystem 类：MySQL 问答系统的主控类，封装了搜索逻辑和交互流程
class MySQLQaSystem:
    # todo __init__ 方法：初始化时创建核心搜索对象（Bm25SoftmaxSearch），它负责真正的检索逻辑
    def __init__(self):
        # todo core_search：核心搜索器实例，基于 BM25 算法 + Softmax 归一化来做相似度匹配
        self.core_search = Bm25SoftmaxSearch()

    # todo search 方法：对外暴露的搜索接口，输入用户问题，返回答案
    def search(self, query:str):
        # todo 记录搜索开始时间，用于计算耗时
        start_time = time.time()
        logger.info(f"开始搜索：{query}")
        # todo 调用核心搜索器的 search 方法进行 BM25 相似度匹配
        # 执行BM25搜索匹配
        answer, _ = self.core_search.search(query)
        if answer:
            logger.info(f"搜索成功，答案长度：{len(answer)}")
        else:
            logger.info(f"搜索失败")
            answer = '【SQL QA系统】未找到答案'
        # todo 计算搜索总耗时（秒），:.3f 表示保留三位小数
        process_time = time.time() - start_time
        logger.info(f"搜索结束，耗时：{process_time:.3f}s")
        return answer

    # todo main 方法：程序启动入口，提供命令行交互界面，用户输入问题后系统返回答案
    def main(self):
        """程序启动入口"""
        try:
            print('\n【欢迎使用MySQL QA系统！（exit退出程序）】')

            # todo 无限循环，持续接收用户输入，直到用户输入 exit 退出
            while True:
                print('\n', '='*20)
                # todo 获取用户输入并去除首尾空格
                query = input("请输入问题：").strip()
                if query == 'exit':
                    print("程序已退出")
                    break
                # todo 调用 search 方法获取答案并打印
                answer = self.search(query)
                print('答案：', answer)
        except Exception as e:
            # todo logger.exception 会同时记录错误信息和完整的调用栈，方便事后排查
            logger.exception(f"【SQL QA系统】程序异常")
            print("程序异常，请检查日志")
        finally:
            # todo finally 块里的代码无论是否异常都会执行，确保数据库连接被关闭
            # 关闭数据库连接
            self.core_search.mysql_client.close()


# todo 当直接运行本文件时（不是被 import），执行以下代码
if __name__ == '__main__':
    # database_init() # 初始化数据库

    # todo 预定义几个测试问题，用于验证系统功能
    q_list = [
        '关联子查询的执行顺序是什么', '归一化算法问题',
        'os.path.dirname() 方法有什么作用？',
        '函数和类区别', '什么是张量',
        '特殊符号如何切割'
    ]

    # todo 创建 MySQLQaSystem 实例并启动命令行交互
    MySQLQaSystem().main() # 程序入口
