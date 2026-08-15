import re  # todo re 是 Python 的正则表达式库，用于字符串匹配和分割
from typing import List, Optional, Any  # todo 类型提示：List 表示列表，Optional 表示可选值，Any 表示任意类型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # todo 导入 LangChain 的递归字符分割器基类


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # todo 这个函数的作用：用分隔符把文本切成一串串小片段
    # todo text 是待切割的文本，separator 是分隔符（可以是正则表达式），keep_separator 表示切完后要不要保留分隔符
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)  # todo 用分隔符切文本，括号()让分隔符本身也被保留在结果里
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]  # todo 把文本片段和它后面的分隔符拼在一起（如 "句子" + "。" -> "句子。"）
            if len(_splits) % 2 == 1:  # todo 如果切出来的片段是奇数个，说明最后还有个没配对的尾巴
                splits += _splits[-1:]  # todo 把最后的尾巴单独加上
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)  # todo 如果不需要保留分隔符，直接切，分隔符丢失
    else:
        splits = list(text)  # todo 如果没有分隔符，就把每个字符单独拆开（一个字符一个片段）
    return [s for s in splits if s != ""]  # todo 过滤掉空字符串，返回所有有效的片段


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    # todo 这个类的作用：专门为中文文本设计的分割器，能把长文章按段落、句子、逗号等逐级切成小段
    # todo 继承自 LangChain 的 RecursiveCharacterTextSplitter，支持递归分割（先按大分隔符切，切不了再换小分隔符）
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        # todo 初始化函数，设置分隔符列表和相关参数
        # todo separators: 自定义的分隔符列表（可选），如果不传就用默认的中文分隔符
        # todo keep_separator: 切完文本后是否保留分隔符（比如保留句号）
        # todo is_separator_regex: 分隔符是否是正则表达式
        super().__init__(keep_separator=keep_separator, **kwargs)  # todo 调用父类的初始化方法
        self._separators = separators or [
            "\n\n",  # todo 第一优先级：两个换行符（段落之间的空行）
            "\n",  # todo 第二优先级：单个换行符（行与行之间）
            "。|！|？",  # todo 第三优先级：中文句号、感叹号、问号（句子结尾）
            "\.\s|\!\s|\?\s",  # todo 第四优先级：英文句号/感叹号/问号后跟空格（英文句子结尾）
            "；|;\s",  # todo 第五优先级：中文/英文分号（从句分隔）
            "，|,\s"  # todo 第六优先级：中文/英文逗号（短语分隔）
        ]
        self._is_separator_regex = is_separator_regex  # todo 记录分隔符是否是正则表达式模式

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        # todo 核心方法：递归地把文本按分隔符逐级切碎，直到每个片段长度不超过 chunk_size
        final_chunks = []  # todo 用来存放最终切好的所有文本块
        # Get appropriate separator to use
        separator = separators[-1]  # todo 默认用最后一个分隔符（最细粒度的）
        new_separators = []  # todo 保存下一轮递归要用的分隔符列表（当前分隔符之后的那些）
        for i, _s in enumerate(separators):  # todo 遍历分隔符列表，找到第一个在文本中能匹配上的
            _separator = _s if self._is_separator_regex else re.escape(_s)  # todo 如果分隔符不是正则，先转义特殊字符
            if _s == "":
                separator = _s  # todo 空字符串表示逐字符切割
                break
            if re.search(_separator, text):  # todo 检查当前分隔符能不能在文本中找到
                separator = _s  # todo 找到了，就用这个分隔符
                new_separators = separators[i + 1:]  # todo 下一轮递归用更细粒度的分隔符
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)  # todo 确定最终使用的分隔符
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)  # todo 用分隔符切文本

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []  # todo 收集长度没超标的短文本片段
        _separator = "" if self._keep_separator else separator  # todo 合并时要加的分隔符
        for s in splits:
            if self._length_function(s) < self._chunk_size:  # todo 如果片段长度没超过 chunk_size 限制
                _good_splits.append(s)  # todo 加入"好片段"列表，先攒着
            else:
                if _good_splits:  # todo 如果之前攒了一些好片段，先把它们合并输出
                    merged_text = self._merge_splits(_good_splits, _separator)  # todo 合并小片段成合适大小的块
                    final_chunks.extend(merged_text)
                    _good_splits = []  # todo 清空好片段列表
                if not new_separators:  # todo 如果没有更细粒度的分隔符了
                    final_chunks.append(s)  # todo 就直接把超长片段原样放入结果（不能再切了）
                else:
                    other_info = self._split_text(s, new_separators)  # todo 还有更细的分隔符，递归切这个超长片段
                    final_chunks.extend(other_info)
        if _good_splits:  # todo 循环结束后，处理最后剩下的一批好片段
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]  # todo 清理：去掉首尾空白，把多余连续换行合并成一个


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,  # todo 切完后保留分隔符（句号、逗号等）
        is_separator_regex=True,  # todo 分隔符是正则表达式模式
        chunk_size=150,  # todo 每个文本块最多 150 个字符
        chunk_overlap=30  # todo 相邻文本块之间重叠 30 个字符（防止语义断在边界上）
    )
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业"缺芯"、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",  # todo 测试用的长文本
        ]
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)  # todo 调用 split_text 把长文本切成多个小块
        for chunk in chunks:
            print(chunk)  # todo 打印每一个切好的文本块
