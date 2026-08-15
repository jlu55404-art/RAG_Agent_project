# todo 这个文件是文本预处理模块，核心功能是"分词"
# todo 什么是分词？就是把一句话切成一个个独立的词语。比如"我爱北京天安门"→["我","爱","北京","天安门"]
# todo 分词是中文自然语言处理的基础步骤，搜索引擎、问答系统都靠分词来理解用户问的是什么

# todo jieba是Python最流行的中文分词库，"结巴"就是"结巴分词"的拼音
import jieba
# todo typing.Generator是类型提示，告诉IDE和开发者这个函数返回一个"生成器"
from typing import Generator
from base import logger


def preprocess_text(
        text: str,
        # todo cut_all=False默认用精确模式（最常用），把句子切成最合理的词语组合
        cut_all: bool = False, # 默认启用精确模式
        # todo is_generator=True返回生成器（省内存，一个一个出词），False返回普通列表（一次性全出来）
        is_generator: bool = True # 返回生成器， False则返回列表
    ) -> Generator | list[str]:
    # todo 这个函数是分词的总入口，支持精确模式和全模式，支持返回生成器或列表
    """对文本进行分词，返回 [] 或 【产生词的生成器】
    1. text.lower() 有英文的先转小写，再分词
    2. 分词返回生成器: jieba.cut(text, cut_all=cut_all)
    3. 分词返回list: jieba.lcut(text, cut_all=False)
    4. 分词模式cut_all参数
        默认False 精准模式，常用最准，计算机 → ['计算机']
        True 全模式，计算机 → ['计算', '算机', '计算机']
    """
    try:
        # todo .lower()把所有英文转成小写，因为"Hello"和"hello"应该被当成同一个词处理
        text = text.lower() # 先转小写
        if is_generator:
            # todo jieba.cut()返回一个生成器——不是一次性把所有词算出来，而是用到一个才产出一个
            # todo 生成器的好处：处理超长文本时不会一次性吃掉大量内存
            result = jieba.cut(text, cut_all=cut_all)
        else:
            # todo jieba.lcut()返回列表——一次性把所有词算好放进列表里，适合短文本
            result = jieba.lcut(text, cut_all=False)
        # logger.info(f"✅ 分词成功：{text}")
        return result
    except Exception as e:
        logger.exception(f"❌ 分词失败")
        return []

if __name__ == '__main__':
    # 👇 测试
    test_text = """
    宙斯手持雷电法杖：我乃万神之王！执掌雷霆，统御诸神！
    玉皇大帝言出法随：你爱谁谁，掏20块钱，我下楼喝碗胡辣汤。
    职业法师刘海柱 我没开挂卢本伟 水晶吊坠周淑怡 宗门圣主蔡徐坤 
    雷电法王杨永信 落樱神斧华盛顿 不准道人海森堡 虐猫狂人薛定谔 
    屏蔽尊者法拉第 万法归一麦克韦 没有子弹燕双鹰 
    法外狂徒张三 无头骑士路易 潮汐海灵袁华 全程围观...
    """

    print("【测试1】👉 默认：精准模式 + 返回生成器")
    # todo preprocess_text()默认返回生成器，list(gen)把生成器里的所有词一次性取出来转成列表
    gen = preprocess_text(test_text)
    assert_msg = '我断言️失败'
    # 【我假设 gen是Generator类型 ✅️】如果错了就异常：assert_msg
    # todo assert isinstance检查gen是不是Generator类型，不是就报错，用来验证函数行为是否符合预期
    assert isinstance(gen, Generator), assert_msg
    print("类型：", type(gen))
    print("转列表结果：", list(gen))

    print("【测试2】👉 精准模式 + 返回列表")
    # todo is_generator=False让jieba用lcut返回列表，适合数据量不大的场景
    lst = preprocess_text(test_text, is_generator=False)
    print("类型：", type(lst))
    print("结果：", lst)

    print("【测试3】👉 全模式 + 返回列表")
    # todo cut_all=True全模式会输出所有可能的词（包括重叠的），"计算机"会拆成"计算""算机""计算机"三种
    lst_all = preprocess_text(test_text,
                              cut_all=True,
                              is_generator=False)
    print("结果：", lst_all)

    print("【测试4】👉 英文自动转小写")
    # todo 验证英文转小写功能："Hello WORLD"变成"hello world"后再分词
    eng_text = "Hello WORLD 你好"
    res = preprocess_text(eng_text, is_generator=False)
    print("结果：", res)

    print("【测试5】👉 空文本/异常输入")
    # todo 测试边界情况：空字符串和None，确保函数不会崩溃，返回空列表
    res1 = preprocess_text("", is_generator=False)
    res2 = preprocess_text(None, is_generator=False)
    assert res1==[] and res2==[], "我断言️失败"
    print("空字符串：", res1)
    print("None：", res2)
