import os
import time

'''将整个项目目录添加到python的导包路径list
实现效果：无所谓当前工作路径在哪，只要程序启动，导包不出错
注意：要在程序入口.py的最开始这么做
'''
import sys
current_file = os.path.abspath(__file__) # 当前文件的绝对路径
current_dir = os.path.dirname(current_file) # 当前文件所在目录 = 项目根目录
sys.path.insert(0, current_dir) # 将项目根目录加入导包路径

# 文档加载分割
from rag_qa.core.process_documents import process_documents
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from base import Config, logger, get_abs_path

# TODO 加入mysql_qa系统
from mysql_qa.core_search import Bm25SoftmaxSearch


def main(query_mode=True, data_path=Config.DATA_DIR):
    '''rag_qa系统入口：
        1. 数据处理 query_mode=False :
            文档加载分割→转向量存向量库
        2. 默认 交互式查询模式 query_mode=True :
            接收用户问题, 调用RAG流程生成答案
    :param query_mode: 运行模式标识, True: 查询模式, False: 数据处理模式
    :param directory_path: 数据处理模式下的文档根目录，相对项目根目录的路径
    :return: 无
    '''
    # TODO 一、数据处理模式(query_mode=False):
    #  对文档进行加载、分割、向量化、存向量库
    if not query_mode:
        logger.info('进入数据处理模式（RAG的离线流程）')
        try:
            # todo 1. 向量存储检索对象 只用存储功能
            vector_store = VectorStore()
        except Exception:
            logger.exception('向量存储检索对象初始化失败')
            return # 失败就提前退出
            # raise '向量存储检索对象初始化失败，检查milvus！'

        # todo 2. 遍历、拼接每个学科的数据目录绝对路径，逐一加载、分割、向量化、存向量库
        total_chunks_added = 0 # [Document]文档块计数器
        # VALID_SOURCES → ["ai", "java", "test", "ops", "bigdata"]
        for sourse in Config.VALID_SOURCES:
            # todo 2.1 拼接各个学科数据文档目录相对于项目根目录的路径
            # data_abs_path ==> rag_qa/data/ai_data
            # data_abs_path = f'{data_path}/{sourse}_data'
            data_abs_path = get_abs_path(f'{data_path}/{sourse}_data')
            logger.info(f'数据处理模式，正在处理目录 {data_abs_path}')
            # todo 2.2 判断学科数据文档目录是否存在
            # 不是绝对路径 或 路径不存在 就跳过
            if not os.path.isabs(
                    data_abs_path) or not os.path.exists(data_abs_path):
                logger.warning(f'{data_abs_path} 不是绝对路径 或 路径不存在')
                # continue
            else:
                # todo 2.3 加载、分割文档，返回子块文档对象列表 [Document]
                try:
                    chunks = process_documents(
                        data_abs_path, # 文档目录路径
                        parent_chunk_size=Config.PARENT_CHUNK_SIZE,  # 父块大小
                        child_chunk_size=Config.CHILD_CHUNK_SIZE,  # 子块大小
                        chunk_overlap=Config.CHUNK_OVERLAP  # 块重叠
                    )
                    if not chunks:
                        logger.warn(f"目录{data_abs_path}未发现有效文档或处理结果为空")
                    # todo 2.4 [Document]转向量存入向量库
                    else:
                        vector_store.add_documents(chunks)
                        total_chunks_added += len(chunks)  # [Document]文档块计数器
                        logger.info(f"成功处理目录 {data_abs_path}，"
                                    f"添加了 {len(chunks)} 个文档块")
                except Exception:
                    logger.exception(f'目录 {data_abs_path} 处理异常')

        logger.info(f'数据处理完成，共添加 {total_chunks_added} 个文档块')

    # TODO 二、交互式查询模式(默认query_mode=True): 接收用户问题, 调用RAG系统生成答案
    else:
        logger.info('进入交互式查询模式（RAG的在线流程）')
        try:
            # todo 1.1 创建RAG系统对象
            rag_system = RAGSystem()
            # todo 1.2 创建mysql_qa系统对象
            bm25_search = Bm25SoftmaxSearch()
        except Exception:
            logger.exception('RAG or FQA系统对象初始化失败')
            print("错误：无法初始化 RAG or FQA系统，无法进入查询模式。")
            return

        # todo 2. 接收用户问题, 调用RAG系统生成答案
        # VALID_SOURCES → ["ai", "java", "test", "ops", "bigdata"]
        valid_sources = Config.VALID_SOURCES
        print("\n欢迎使用 EduRAG 交互式查询系统！")
        while True:
            query = input("\n请输入您的问题（或输入 'exit' 退出系统）：")
            # todo 系统退出逻辑
            if query.lower() == 'exit':
                print("感谢使用 EduRAG 交互式查询系统！")
                return # 退出系统
            # todo 3. 获取用户选择的学科类别
            source_filter_input = input( # 去空格、转小写
                f"请输入学科类别{valid_sources}，直接回车默认全部：").strip().lower()
            source_filter = None  # 学科类别，默认不过滤
            if source_filter_input in valid_sources:
                source_filter = source_filter_input
            else:
                logger.warn(f'无效的学科类别 {source_filter_input}')
                print(f"输入的学科类别 {source_filter_input} 无效，将使用默认值。")

            # 计算耗时
            start_time = time.time()
            # TODO ==============  调用mysql_qa系统生成答案
            try:
                print('正在调用mysql_qa系统生成答案...')
                answer, need_rag = bm25_search.search(query)
                # 如果不需要RAG系统，就提前跳过当前循环
                if not need_rag:
                    print("-" * 30)
                    print(f"问题: {query}")
                    print(f'学科: {source_filter}')
                    print(f"回答: {answer}")
                    print(f"耗时: {time.time() - start_time}秒")
                    print("-" * 30)
                    continue
            except Exception:
                logger.exception(f'问题 {query} '
                                 f'mysql_qa系统处理异常')

            # todo 4. 调用RAG系统生成答案
            try:
                answer = rag_system.generate_answer(query, source_filter)
                print("-" * 30)
                print(f"问题: {query}")
                print(f'学科: {source_filter}')
                print(f"回答: {answer}")
                print(f"耗时: {time.time() - start_time}秒")
                print("-" * 30)
            except Exception:
                logger.exception(f'问题 {query} RAG系统处理异常')
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员"
                      f"{Config.CUSTOMER_SERVICE_PHONE}。\n")



'''命令行模式：通过命令行参数控制'''
def cmd_run():
    import argparse
    print('➡️ Edu RAG 命令行模式：')

    # todo 1. 自定义程序介绍和说明
    description = """EduRAG v7.9 启动说明：
    1. 默认进入RAG问答模式：
        python ./main.py
    2. 查看命令菜单
        python ./main.py --h
        python ./main.py --help 
    3. 进入RAG数据处理模式，对默认目录中的数据文件进行处理 
        python ./main.py --data
    4. 进入RAG数据处理模式，对指定目录中的数据文件进行处理 
        python ./main.py --data --dir 数据文件的相对或绝对路径
    """
    epilog = f'感谢使用 EduRAG v7.9！如有问题请联系管理员: {Config.CUSTOMER_SERVICE_PHONE}'

    # todo 2. 创建参数解析器，实例化ArgumentParser对象，设置"描述信息"
    parser = argparse.ArgumentParser(
        description=description, # 描述信息
        epilog=epilog, # 额外说明信息
        # 按照描述信息的格式进行显示
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # todo 3. 添加数据处理模式的参数
    #   --data 表示 进入RAG离线模式
    parser.add_argument(
        '--data', # 设置的参数
        action='store_true', # 设置参数为布尔值
        help='进入RAG数据处理模式(离线模式)'
    )
    '''参数 action='store_true'
    1. 当命令行出现 --data → 自动将 args.data 设为 True
    2. 当命令行未出现 --data → 自动将 args.data 设为 False
    '''

    #  todo 4. 添加数据处理模式的参数
    #   --dir 表示 指定数据文件的相对或绝对路径
    parser.add_argument(
        '--dir', # 设置的参数
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=Config.DATA_DIR, # 默认数据目录的绝对路径
        help='在--data模式下 输入数据目录的路径 进行数据处理'
    )
    '''保障相对路径和绝对路径都支持（默认仅支持绝对路径）
    type=lambda p: os.path.abspath(os.path.expanduser(p))
    1. os.path.abspath() 
        先调用 os.path.isabs() 进行判断。
        如果已经是绝对路径，它直接返回；
        如果是相对路径，它会基于当前工作目录os.getcwd()自动拼接成绝对路径
    2. os.path.expanduser()
        将路径字符串中的波浪号（~）替换为当前用户家目录的绝对路径
    '''

    # todo 5. 解析命令行参数
    args = parser.parse_args()
    print(args.data, args.dir)

    # todo 6. 执行main函数
    main(
        # 有命令行参数时，args.data 为 True，则 query_mode 为 False
        query_mode=(not args.data), # False就是进入离线模式
        data_path=args.dir # 命令行指定的数据目录路径
    )



if __name__ == '__main__':
    '''数据处理模式'''
    # milvus必须提前创建好数据库 库名在配置文件中 database_name = management_qa
    # print('➡️ Edu RAG 数据处理模式：')
    # main(query_mode=False)


    '''运行查询模式'''
    # print('➡️ Edu RAG QA查询模式：')
    # main(query_mode=True)

    '''命令行模式'''
    cmd_run()






