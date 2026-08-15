# todo 这个文件是基于MySQL的FAQ系统的"数据库操作模块"
# todo 它就像一个"仓库管理员"，负责跟MySQL数据库打交道：建表、存数据、查数据、自动重连等
# todo 整个项目的FAQ问答数据都存在MySQL里，通过这个模块来读写

# ===========================================
# 基于 mysql FQA系统的 mysql操作模块
# ===========================================
# todo pymysql是Python连接MySQL的驱动库，相当于"电话线"，让Python程序能跟MySQL数据库通话
import pymysql
# todo pandas是Python最常用的数据分析库，这里用它来读取CSV文件数据
import pandas as pd

# todo base是本项目的公共模块，Config存配置、logger记日志、get_abs_path转绝对路径
from base import Config, logger, get_abs_path


class MysqlClient:
    # todo MysqlClient类封装了所有MySQL操作，包括增删改查
    def __init__(self):
        # todo __init__是"构造函数"，创建对象时自动执行，负责建立数据库连接
        try:
            # 连接对象
            # todo pymysql.connect()与MySQL建立TCP连接，下面这些参数分别是数据库地址、端口、用户名、密码、库名、字符集
            self.conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                # todo utf8mb4是MySQL支持最全的字符集，能存emoji表情和所有中文
                charset='utf8mb4'
            )
            # todo 游标(cursor)相当于"鼠标指针"，通过它来执行SQL语句并获取结果
            self.cur = self.conn.cursor() # 游标对象
            self.table_name = Config.MYSQL_TABLE  # 表名
            logger.info(
                f"✅ 连接数据库成功："
                f"{Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DB}，"
                f"username:{Config.MYSQL_USER}")
        except Exception as e:
            logger.exception(f"❌ 连接数据库失败")
            raise e

    def create_table(self):
        # todo 创建表的方法：先删除旧表（如果存在），再建新表，相当于"推倒重建"
        """先删、再建表"""
        try:
            # todo DROP TABLE IF EXISTS：如果表存在就删掉，不存在也不报错
            self.cur.execute(
                f"DROP TABLE IF EXISTS `{self.table_name}` ")

            # todo 建表SQL：id是自增主键（每插入一行自动+1），subject_name是学科名，question和answer是问答对
            create_sql = f"""
                CREATE TABLE `{self.table_name}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    subject_name VARCHAR(20),
                    question VARCHAR(1000),
                    answer VARCHAR(1000)
                ) ENGINE=InnoDB;
            """
            self.cur.execute(create_sql)
            # todo commit()提交事务——MySQL操作默认是"先记账不付钱"，commit才是真正执行
            self.conn.commit() # 提交
            logger.info(f"✅ 创建表成功：{self.table_name}")
        except Exception as e:
            logger.exception(f"❌ 创建表失败")
            raise e

    def check_and_reconnect(self):
        # todo MySQL连接可能因为网络波动、长时间闲置而断开，这个方法检查连接状态，断了就重连
        """mysql连接会断开（网络波动、超时）
        检查并重连"""
        try:
            # todo ping()是MySQL自带的心跳检测，发一个"你还活着吗"的包
            self.conn.ping()
        except:
            # todo 如果ping失败说明连接断开了，先关闭旧连接，再重新初始化
            self.close()
            self.__init__() # 重新初始化进行连接

    def insert_data(self, csv_path):
        # todo 从CSV文件读数据，批量插入到MySQL表中
        """插入csv文件测试数据"""
        self.check_and_reconnect() # 重连
        try:
            # 读取数据
            # todo pd.read_csv()读取CSV文件为DataFrame（表格数据结构）
            df = pd.read_csv(csv_path)
            # todo values.tolist()把DataFrame转成二维列表，每个元素是一行数据
            data = df.values.tolist() # [[一条数据], [], ...]
            # todo %s是占位符，executemany会自动把data里的每一行替换进去，防止SQL注入
            insert_sql = (f"INSERT INTO `{self.table_name}` "
                          f"(subject_name, question, answer) "
                          f"VALUES (%s, %s, %s)")
            # 批量插入
            # todo executemany()批量执行SQL，比一条一条插入快很多
            self.cur.executemany(insert_sql, data)
            self.conn.commit()
            logger.info(f"✅ 插入数据成功：{csv_path}")
        except Exception as e:
            logger.exception(f"❌ 插入数据失败")
            raise e

    def fetch_all_questions(self, table_name=Config.MYSQL_TABLE):
        # todo 查询表中所有问题，返回给调用方（比如用于构建FAQ索引）
        """获取所有问题"""
        self.check_and_reconnect()  # 重连
        try:
            sql = f"SELECT question FROM `{table_name}`"
            self.cur.execute(sql)
            # todo fetchall()把所有查询结果一次性取出来，返回元组列表
            results = self.cur.fetchall() # [(问题1str), (问题2str), ...]
            logger.info(f"✅ 获取所有问题成功：{len(results)}")
            return results
        except Exception as e:
            logger.exception(f"❌ 获取所有问题失败")
            raise e

    def fetch_answer(self, question:str):
        # todo 根据问题内容精确匹配答案，类似于"输入问题→找到对应答案"的FAQ查找
        """根据问题字符串，获取对应的答案"""
        self.check_and_reconnect()  # 重连
        try:
            '''表名没有参数化
            sql = f"""SELECT answer 
            FROM `{Config.MYSQL_TABLE}` 
            WHERE question = %s"""
            self.cur.execute(sql, (question,)) # sql参数化
            '''
            # TODO 表名参数化
            # todo escape_string()对表名字符串做安全转义，防止特殊字符导致SQL报错
            safe_table = self.conn.escape_string(Config.MYSQL_TABLE)
            sql = f"""SELECT answer 
            FROM `{safe_table}` 
            WHERE question = %s"""
            # todo (question,)是参数化查询：把用户输入和SQL语句分开传，防止SQL注入攻击
            self.cur.execute(sql, (question,)) # TODO 参数化
            # todo fetchone()只取第一条结果，因为问题应该唯一
            result = self.cur.fetchone() # (答案,)
            logger.info(f"✅ 获取答案成功：{question}")
            # todo result[0]取元组的第一个元素（答案字符串），没找到就返回None
            return result[0] if result else None
        except Exception as e:
            logger.exception(f"❌ 获取答案失败")
            raise e

    def close(self):
        # todo 关闭游标和连接，释放数据库资源（用完要关，不然会占着连接不释放）
        """关闭连接"""
        self.cur.close()
        self.conn.close()

    def __del__(self):
        # todo __del__是"析构函数"，对象被垃圾回收时自动调用，确保连接被关闭
        """对象回收时自动触发、静默处理"""
        try:
            self.close()
        except:
            pass

# 数据库初始化：建库、建表、插入测试数据
def database_init(
        csv_file_path='mysql_qa/data/JP学科知识问答.csv'
    ):
    # todo 数据库初始化函数：创建数据库→创建表→插入CSV测试数据，一键搞定
    # 数据库client
    client = MysqlClient()
    # 创建数据库
    # todo CREATE DATABASE IF NOT EXISTS：如果数据库不存在就创建，存在就跳过
    create_db_sql = """CREATE DATABASE 
        IF NOT EXISTS subjects_kg"""
    client.cur.execute(create_db_sql)
    client.conn.commit()

    # 创建表
    client.create_table()

    # 插入数据
    # todo get_abs_path把相对路径转成绝对路径，确保不管从哪儿运行都能找到文件
    file_abs_path = get_abs_path(csv_file_path)
    client.insert_data(file_abs_path)


if __name__ == '__main__':
    # TODO 项目数据库初始化
    # todo 直接运行这个文件就会执行数据库初始化，把测试数据灌进去
    database_init()

    # # 拓展： \U会报错，需要添加r就不报错了
    # #   让\就是自己这个反斜杠，而不是转义
    # pd.read_csv(r'C:\Users\xxx\Desktop\Stady\郑州AI2期\02-EduRAG项目\mysql_qa\data\JP学科知识问答.csv')
