# todo 导入 os 模块，用来操作文件路径（比如获取当前文件的目录、拼接路径等）
import os

'''将整个项目目录添加到python的导包路径list
实现效果：无所谓当前工作路径在哪，只要程序启动，导包不出错
注意：要在程序入口.py的最开始这么做
'''
# todo 导入 sys 模块，sys.path 是 Python 搜索模块的路径列表，往里面加路径就能让 Python 找到我们的项目包
import sys
# todo 下面3行是"自动找项目根目录"的黑魔法：
# todo __file__ 是当前文件的路径，os.path.abspath 把它转成绝对路径（Windows 下带盘符那种）
current_file = os.path.abspath(__file__) # 当前文件的绝对路径
# todo os.path.dirname 取上一级目录，第1次拿到 rag_qa 目录
current_dir = os.path.dirname(current_file) # 当前文件所在目录
# todo 再往上取一级，就是项目根目录 RAG_EDU_FQA
project_root = os.path.dirname(current_dir) # 工程根目录
# todo 把项目根目录加到 Python 的搜索路径里，这样不管从哪个目录运行这个脚本，都能找到项目里的其他模块
sys.path.append(project_root)

# 文档加载分割
# todo process_documents：负责把文档（pdf、word、txt等）读进来、切成小块的函数
from rag_qa.core.process_documents import process_documents
# todo VectorStore：向量存储类，把文档块转成向量（一串数字），存到 Milvus 向量数据库里，后面检索用
from rag_qa.core.vector_store import VectorStore
# todo RAGSystem：RAG 的核心类，负责"检索相关文档 + 喂给大模型生成答案"的完整流程
from rag_qa.core.rag_system import RAGSystem
# todo Config：项目配置文件类；logger：日志记录器，用来打日志方便排查问题；get_abs_path：把相对路径转成绝对路径
from base import Config, logger, get_abs_path


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
    # todo 这是一个"双模式"函数：参数 query_mode 决定走哪条分支
    # todo query_mode=True（默认）：在线问答模式，用户问问题→系统检索+生成答案
    # todo query_mode=False：离线模式，把文档处理好存到向量库，供问答模式用
    # TODO 一、数据处理模式(query_mode=False):
    #  对文档进行加载、分割、向量化、存向量库
    if not query_mode:
        logger.info('进入数据处理模式（RAG的离线流程）')
        try:
            # todo 1. 向量存储检索对象 只用存储功能
            # todo VectorStore 初始化时会连接 Milvus 向量数据库，如果连不上就报错退出
            vector_store = VectorStore()
        except Exception:
            logger.exception('向量存储检索对象初始化失败')
            # todo return 直接退出函数，因为向量库连不上，后续的存向量操作都没法做
            return # 失败就提前退出
            # raise '向量存储检索对象初始化失败，检查milvus！'

        # todo 2. 遍历、拼接每个学科的数据目录绝对路径，逐一加载、分割、向量化、存向量库
        # todo 这个计数器用来统计一共处理了多少个文档块，最后打印汇总用
        total_chunks_added = 0 # [Document]文档块计数器
        # VALID_SOURCES → ["ai", "java", "test", "ops", "bigdata"]
        # todo 循环遍历每个学科目录（如 ai_data、java_data 等），逐个处理
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
                # todo process_documents 做了三件事：①读取目录下所有文档 ②按父子块策略切分 ③返回切好的文档块
                try:
                    chunks = process_documents(
                        data_abs_path, # 文档目录路径
                        parent_chunk_size=Config.PARENT_CHUNK_SIZE,  # 父块大小（大块，给大模型看完整上下文用的）
                        child_chunk_size=Config.CHILD_CHUNK_SIZE,  # 子块大小（小块，用来做精确向量检索的）
                        chunk_overlap=Config.CHUNK_OVERLAP  # 块之间重叠的字符数，防止关键信息被切断在边界
                    )
                    if not chunks:
                        logger.warn(f"目录{data_abs_path}未发现有效文档或处理结果为空")
                    # todo 2.4 [Document]转向量存入向量库
                    # todo add_documents 会把文档块内容转成向量（一串数字），然后存到 Milvus 里
                    else:
                        vector_store.add_documents(chunks)
                        total_chunks_added += len(chunks)  # [Document]文档块计数器
                        logger.info(f"成功处理目录 {data_abs_path}，"
                                    f"添加了 {len(chunks)} 个文档块")
                except Exception:
                    logger.exception(f'目录 {data_abs_path} 处理异常')

        logger.info(f'数据处理完成，共添加 {total_chunks_added} 个文档块')

    # TODO 二、交互式查询模式(默认query_mode=True): 接收用户问题, 调用RAG系统生成答案
    # todo 这是"在线模式"：用户输入问题→系统从向量库检索相关文档→把文档+问题一起给大模型→生成回答
    else:
        logger.info('进入交互式查询模式（RAG的在线流程）')
        try:
            # todo 1. 创建RAG系统对象
            # todo RAGSystem 初始化时会连接 Milvus 向量库、加载嵌入模型和大语言模型
            rag_system = RAGSystem()
        except Exception:
            logger.exception('RAG系统对象初始化失败')
            print("错误：无法初始化 RAG 系统，无法进入查询模式。")
            return

        # todo 2. 接收用户问题, 调用RAG系统生成答案
        # VALID_SOURCES → ["ai", "java", "test", "ops", "bigdata"]
        # todo valid_sources 是学科类别列表，用户可以选择只看某个学科的结果（如只看 ai 相关文档）
        valid_sources = Config.VALID_SOURCES
        print("\n欢迎使用 EduRAG 交互式查询系统！")
        # todo 用一个无限循环让用户反复问问题，输入 exit 才退出
        while True:
            query = input("\n请输入您的问题（或输入 'exit' 退出系统）：")
            # todo 系统退出逻辑
            if query.lower() == 'exit':
                print("感谢使用 EduRAG 交互式查询系统！")
                return # 退出系统
            # todo 3. 获取用户选择的学科类别
            # todo 用户输入学科类别（如 ai、java），用来过滤检索范围，只看某个学科的文档
            source_filter_input = input( # 去空格、转小写
                f"请输入学科类别{valid_sources}，直接回车默认全部：").strip().lower()
            source_filter = None  # 学科类别，默认不过滤（None 表示查所有学科）
            if source_filter_input in valid_sources:
                source_filter = source_filter_input
            else:
                logger.warn(f'无效的学科类别 {source_filter_input}')
                print(f"输入的学科类别 {source_filter_input} 无效，将使用默认值。")

            # todo 4. 调用RAG系统生成答案
            # todo generate_answer 是核心：①向量检索相关文档 ②把文档+问题拼成提示词 ③发给大模型 ④返回答案
            try:
                answer = rag_system.generate_answer(query, source_filter)
                print("-" * 30)
                print(f"问题: {query}")
                print(f'学科: {source_filter}')
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception:
                logger.exception(f'问题 {query} RAG系统处理异常')
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员"
                      f"{Config.CUSTOMER_SERVICE_PHONE}。\n")


'''命令行模式：通过命令行参数控制'''
# todo cmd_run 函数让你可以通过命令行参数（如 --data、--dir）来控制程序行为
# todo 比如 python main.py --data --dir ./data 就会进入离线模式处理指定目录的数据
def cmd_run():
    # todo argparse 是 Python 自带的命令行参数解析库，帮你把用户打的命令行参数解析成程序能用的变量
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
    # todo 这行是 Python 的标准写法：当直接运行这个文件时（不是被 import 时），执行下面的代码
    # todo __name__ 是 Python 内置变量，直接运行脚本时值为 '__main__'，被导入时值为模块名
    '''数据处理模式'''
    # milvus必须提前创建好数据库 库名在配置文件中 database_name = management_qa
    # print('➡️ Edu RAG 数据处理模式：')
    # main(query_mode=False)


    '''运行查询模式'''
    # print('➡️ Edu RAG QA查询模式：')
    # main(query_mode=True)

    '''命令行模式'''
    cmd_run()
