
# todo 这个文件是Redis缓存客户端模块
# todo Redis是一个超快的内存数据库，就像程序的"速记本"——把常用数据暂存在内存里，下次取的时候不用再查数据库或调API，秒级返回
# todo 在这个项目里，Redis用来缓存FAQ的答案，一个问题第一次查询后存到Redis，下次同样问题直接返回缓存结果

# todo redis库是Python连接Redis的官方驱动
import redis, json
from base import Config, logger


class RedisClient:
    # todo RedisClient封装了所有Redis操作：存数据、取数据、缓存问答、清空缓存
    def __init__(self):
        # todo 构造方法：创建Redis连接
        try:
            # todo redis.Redis()连接Redis服务器，参数包括地址、端口、密码、数据库编号
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD,
                db=Config.REDIS_DB,
                # todo decode_responses=True让Redis返回普通字符串而不是bytes字节串，否则中文会显示成 b'\xe4\xbd\xa0\xe5\xa5\xbd'
                decode_responses=True # 默认是bytes，设置成True，则返回字符串
            )
            logger.info(f"✅ Redis连接成功："
                        f"{Config.REDIS_HOST}:"
                        f"{Config.REDIS_PORT}")
        except Exception as e:
            logger.exception(f"❌ Redis连接失败")
            raise e

    def set_data(self, key:str, value):
        # todo 存数据到Redis：key是键（像一个标签），value是值（任何Python数据，会被转成JSON字符串存储）
        """存储数据到redis
        1. Redis不能存Python稍微复杂点的数据类型→必须先json.dumps转字符串
        2. json.dumps(xxx, ensure_ascii=False) →
            2.1 字典dict ➔ JSON对象{}。字典的键必须是字符串
            2.2 列表list/元组tuple ➔ JSON数组[]
            2.3 字符串str ➔ JSON字符串
            2.4 数字int/float) ➔ JSON数字，支持正负数、小数等
            2.5 布尔值True/False ➔ JSON小写布尔值true/false
            2.6 空值None ➔ JSON空值null
        3. 参数ensure_ascii=False，让中文不乱码
            关闭默认的"非ASCII字符自动转义"
                json.dumps({"name": "张三"}) →
                    '{"name": "\u5f20\u4e09"}'
                json.dumps({"name": "张三"}, ensure_ascii=False) →
                    '{"name": "张三"}'
        4. 可以用 json.loads(xxx) 还原
        """
        try:
            # todo self.client.set()是Redis的"set"命令：存一个键值对
            self.client.set(
                name=key,
                # todo json.dumps()把Python对象转成JSON字符串，ensure_ascii=False让中文正常显示
                value=json.dumps(value, ensure_ascii=False),
                # todo ex参数设置过期时间（秒），超过这个时间数据自动删除，防止缓存永远不更新
                ex=Config.REDIS_TTL # 过期时间，秒
            )
            logger.info(f"✅ 存储数据成功：{key}")
        except Exception as e:
            logger.exception(f"❌ 存储数据失败")
            raise e


    def get_data(self, key:str):
        # todo 从Redis取数据：用key查找对应的值，找到后用json.loads把JSON字符串还原成Python对象
        """获取数据"""
        try:
            # todo get()根据key获取值，如果key不存在返回None
            value = self.client.get(key)
            if value:
                logger.info(f"✅ 获取数据成功：{key}")
                # todo json.loads()是json.dumps()的逆操作，把JSON字符串变回Python对象
                return json.loads(value)
            else:
                logger.info(f"❌ 获取数据失败：{key}")
                return None
        except Exception as e:
            logger.exception(f"❌ 获取数据失败")
            raise e

    def set_answer(self, query:str, answer:str):
        # todo 缓存一个问题的答案：key格式是 "answer:问题内容"，value就是答案字符串
        """存储一个问题的答案到redis"""
        try:
            self.client.set(
                # todo 用 "answer:" 前缀来区分缓存类型，方便管理和查找
                name=f'answer:{query}',
                value=answer,
                ex=Config.REDIS_TTL # 过期时间，秒
            )
            logger.info(f"✅ 存储答案成功：{query}")
        except Exception as e:
            logger.exception(f"❌ 存储答案失败")
            raise e

    def get_answer(self, query:str):
        # todo 查找缓存中有没有某个问题的答案，有就直接返回（缓存命中），没有就返回None（缓存未命中，需要去查数据库或调大模型）
        """获取一个问题的答案"""
        try:
            answer = self.client.get(f'answer:{query}')
            if answer:
                logger.info(f"✅ 获取答案成功：{query}")
                return answer
            else:
                logger.info(f"❌ 获取答案失败：{query}")
                return None
        except Exception as e:
            logger.exception(f"❌ 获取答案失败")
            raise e

    def clear_all_data(self):
        # todo 清空当前Redis数据库的所有数据，相当于"一键重置"
        """清空缓存数据"""
        try:
            # todo flushdb()只清空当前数据库（db=Config.REDIS_DB指定的那个），不影响其他数据库
            self.client.flushdb()
            logger.info(f"✅ 清空缓存数据成功")
        except Exception as e:
            logger.exception(f"❌ 清空缓存数据失败")
            raise e

    def __del__(self):
        # todo 对象销毁时自动关闭Redis连接，释放资源
        try:
            self.client.close()
        except:
            pass


if __name__ == '__main__':
    # 测试
    # todo 下面的测试代码展示了如何使用RedisClient
    RedisClient().clear_all_data()
    RedisClient().set_answer('1+1', '2')
    print(RedisClient().get_answer('1+1'))
    RedisClient().set_data('1+1', '2')
    print(RedisClient().get_data('1+1'))
