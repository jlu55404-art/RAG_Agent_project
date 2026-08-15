# -*- coding: utf-8 -*-
"""
冒烟测试 - 快速验证RAG系统核心功能
先检测各服务连通性，不可用的服务跳过对应测试
"""
import sys
import time
import socket
from datetime import datetime

sys.path.insert(0, r"D:\RAG_project")


class SmokeTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.skipped = 0
        self.results = []

        self.mysql_ok = False
        self.redis_ok = False
        self.milvus_ok = False

    def log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "🔵",
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️",
            "TEST": "🧪",
            "SKIP": "⏭️"
        }.get(level, "•")
        print(f"[{timestamp}] {prefix} {message}")

    def add_result(self, test_name, status, details="", duration=0):
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details,
            "duration": duration
        })
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        elif status == "WARN":
            self.warnings += 1
        elif status == "SKIP":
            self.skipped += 1

    def check_connectivity(self):
        """预检各服务连通性"""
        from base.config import Config
        self.config = Config

        print("\n📡 服务连通性预检")
        print("-" * 50)

        # 1. MySQL
        try:
            sock = socket.create_connection(
                (Config.MYSQL_HOST, Config.MYSQL_PORT), timeout=3
            )
            sock.close()
            self.mysql_ok = True
            self.log("PASS", f"MySQL {Config.MYSQL_HOST}:{Config.MYSQL_PORT} 可达")
        except Exception as e:
            self.mysql_ok = False
            self.log("FAIL", f"MySQL {Config.MYSQL_HOST}:{Config.MYSQL_PORT} 不可达 ({e})")

        # 2. Redis
        try:
            sock = socket.create_connection(
                (Config.REDIS_HOST, Config.REDIS_PORT), timeout=3
            )
            sock.close()
            self.redis_ok = True
            self.log("PASS", f"Redis {Config.REDIS_HOST}:{Config.REDIS_PORT} 可达")
        except Exception as e:
            self.redis_ok = False
            self.log("FAIL", f"Redis {Config.REDIS_HOST}:{Config.REDIS_PORT} 不可达 ({e})")

        # 3. Milvus
        try:
            sock = socket.create_connection(
                (Config.MILVUS_HOST, Config.MILVUS_PORT), timeout=3
            )
            sock.close()
            self.milvus_ok = True
            self.log("PASS", f"Milvus {Config.MILVUS_HOST}:{Config.MILVUS_PORT} 可达")
        except Exception as e:
            self.milvus_ok = False
            self.log("FAIL", f"Milvus {Config.MILVUS_HOST}:{Config.MILVUS_PORT} 不可达 ({e})")

        print("-" * 50)
        available = sum([self.mysql_ok, self.redis_ok, self.milvus_ok])
        print(f"可用服务: {available}/3")

        if not self.mysql_ok:
            print("  ⚠️ FAQ匹配测试将被跳过")
        if not self.milvus_ok:
            print("  ⚠️ RAG检索/端到端/边界测试将被跳过")

        return available > 0

    def test_faq_matching(self):
        """测试FAQ匹配功能"""
        self.log("TEST", "\n" + "=" * 60)
        self.log("TEST", "测试1: FAQ匹配功能")
        self.log("TEST", "=" * 60)

        if not self.mysql_ok or not self.redis_ok:
            self.log("SKIP", "MySQL或Redis不可达，跳过FAQ匹配测试")
            self.add_result("FAQ匹配功能", "SKIP", "MySQL/Redis不可达")
            return

        try:
            from mysql_qa.core_search import Bm25SoftmaxSearch
            bm25 = Bm25SoftmaxSearch()
            self.log("INFO", "BM25搜索初始化成功")

            # 测试用例
            test_cases = [
                ("什么是管理学", True),
                ("泰勒的科学管理理论", True),
                ("马斯洛需求层次", True),
                ("管理的四大职能", True),
                ("组织设计原则", True),
            ]

            threshold = self.config.BM25_THRESHOLD
            self.log("INFO", f"BM25阈值: {threshold}")

            for question, expect_match in test_cases:
                start_time = time.time()
                # search() 返回 (answer, need_rag)
                answer, need_rag = bm25.search(question, threshold=threshold)
                duration = time.time() - start_time

                if not need_rag and answer:
                    self.log("PASS", f"'{question}' → 匹配成功 (答案{len(answer)}字, {duration:.2f}s)")
                    self.add_result(f"FAQ: {question}", "PASS", f"答案{len(answer)}字", duration)
                elif need_rag:
                    if expect_match:
                        self.log("FAIL", f"'{question}' → 未匹配（期望命中FAQ）")
                        self.add_result(f"FAQ: {question}", "FAIL", "未命中FAQ", duration)
                    else:
                        self.log("PASS", f"'{question}' → 正确未命中FAQ，需要RAG ({duration:.2f}s)")
                        self.add_result(f"FAQ: {question}", "PASS", "正确走RAG", duration)
                else:
                    self.log("WARN", f"'{question}' → 返回空答案")
                    self.add_result(f"FAQ: {question}", "WARN", "空答案", duration)

        except Exception as e:
            self.log("FAIL", f"FAQ匹配测试异常: {str(e)}")
            self.add_result("FAQ匹配功能", "FAIL", str(e))

    def test_rag_retrieval(self):
        """测试RAG检索功能"""
        self.log("TEST", "\n" + "=" * 60)
        self.log("TEST", "测试2: RAG向量检索功能")
        self.log("TEST", "=" * 60)

        if not self.milvus_ok:
            self.log("SKIP", "Milvus不可达，跳过RAG检索测试")
            self.add_result("RAG检索功能", "SKIP", "Milvus不可达")
            return

        try:
            from rag_qa.core.rag_system import RAGSystem
            rag = RAGSystem()  # 不接受参数
            self.log("INFO", "RAG系统初始化成功")

            test_cases = [
                {
                    "question": "泰勒的科学管理理论主要内容是什么",
                    "expected_keywords": ["泰勒", "科学管理", "工作定额"]
                },
                {
                    "question": "那个说人天生懒惰的理论叫什么",
                    "expected_keywords": ["麦格雷戈", "X理论", "人性"]
                },
                {
                    "question": "泰勒和法约尔的管理理论有什么区别",
                    "expected_keywords": ["泰勒", "法约尔"]
                },
                {
                    "question": "如何提高管理效率",
                    "expected_keywords": ["效率", "管理"]
                }
            ]

            for case in test_cases:
                question = case["question"]
                expected_keywords = case["expected_keywords"]

                self.log("INFO", f"\n测试问题: '{question}'")
                start_time = time.time()

                try:
                    # 用vector_store的hybrid_search_with_rerank做检索
                    docs = rag.vector_store.hybrid_search_with_rerank(
                        question,
                        source_filter="management"
                    )
                    duration = time.time() - start_time

                    if not docs or len(docs) == 0:
                        self.log("FAIL", f"检索失败: 未返回任何文档")
                        self.add_result(f"RAG: {question}", "FAIL", "无文档", duration)
                        continue

                    if len(docs) < 3:
                        self.log("WARN", f"返回文档较少: {len(docs)}个 (期望>=3)")

                    combined_text = " ".join([doc.page_content for doc in docs])
                    found_keywords = [kw for kw in expected_keywords if kw in combined_text]
                    keyword_ratio = len(found_keywords) / len(expected_keywords)

                    if keyword_ratio >= 0.5:
                        self.log("PASS", f"检索成功: {len(docs)}个文档, 关键词: {found_keywords} ({duration:.2f}s)")
                        self.add_result(f"RAG: {question}", "PASS", f"{len(docs)}文档, {keyword_ratio:.0%}关键词", duration)
                    else:
                        self.log("WARN", f"关键词匹配不足: {found_keywords}/{expected_keywords} ({keyword_ratio:.0%})")
                        self.add_result(f"RAG: {question}", "WARN", f"关键词匹配{keyword_ratio:.0%}", duration)

                    for i, doc in enumerate(docs[:3]):
                        preview = doc.page_content[:100].replace("\n", " ")
                        source = doc.metadata.get("file_name", "未知")
                        page = doc.metadata.get("page_start", "?")
                        self.log("INFO", f"  文档{i+1}: [{source} p{page}] {preview}...")

                except Exception as e:
                    duration = time.time() - start_time
                    self.log("FAIL", f"检索异常: {str(e)}")
                    self.add_result(f"RAG: {question}", "FAIL", str(e), duration)

        except Exception as e:
            self.log("FAIL", f"RAG系统初始化失败: {str(e)}")
            self.add_result("RAG系统初始化", "FAIL", str(e))

    def test_end_to_end(self):
        """端到端测试：完整问答流程"""
        self.log("TEST", "\n" + "=" * 60)
        self.log("TEST", "测试3: 端到端问答流程")
        self.log("TEST", "=" * 60)

        if not self.milvus_ok:
            self.log("SKIP", "Milvus不可达，跳过低到端测试")
            self.add_result("端到端测试", "SKIP", "Milvus不可达")
            return

        try:
            from rag_qa.core.rag_system import RAGSystem
            rag = RAGSystem()

            test_questions = [
                "管理的四大职能是什么",
                "泰勒的科学管理理论",
            ]

            for question in test_questions:
                self.log("INFO", f"\n测试问题: '{question}'")
                start_time = time.time()

                try:
                    # generate_answer(query, source_filter, user_id, session_id)
                    result = rag.generate_answer(
                        query=question,
                        source_filter="management"
                    )
                    duration = time.time() - start_time

                    # result 可能是字符串或dict
                    if isinstance(result, dict):
                        answer = result.get("answer", "")
                        sources = result.get("sources", [])
                    elif isinstance(result, str):
                        answer = result
                        sources = []
                    elif isinstance(result, tuple):
                        answer = result[0] if result[0] else ""
                        sources = result[1] if len(result) > 1 else []
                    else:
                        answer = str(result)
                        sources = []

                    if not answer or len(answer) < 20:
                        self.log("FAIL", f"答案过短: {len(answer)}字符")
                        self.add_result(f"E2E: {question}", "FAIL", f"答案{len(answer)}字", duration)
                        continue

                    if "超出当前知识库范围" in answer or "人工客服" in answer:
                        self.log("WARN", f"系统拒绝回答: {answer[:100]}...")
                        self.add_result(f"E2E: {question}", "WARN", "系统拒绝回答", duration)
                        continue

                    self.log("PASS", f"答案生成成功: {len(answer)}字符 ({duration:.2f}s)")
                    self.add_result(f"E2E: {question}", "PASS", f"{len(answer)}字", duration)

                    answer_preview = answer[:200].replace("\n", " ")
                    self.log("INFO", f"  答案预览: {answer_preview}...")

                except Exception as e:
                    duration = time.time() - start_time
                    self.log("FAIL", f"问答异常: {str(e)}")
                    self.add_result(f"E2E: {question}", "FAIL", str(e), duration)

        except Exception as e:
            self.log("FAIL", f"端到端测试初始化失败: {str(e)}")
            self.add_result("端到端测试", "FAIL", str(e))

    def test_boundary_cases(self):
        """测试边界情况"""
        self.log("TEST", "\n" + "=" * 60)
        self.log("TEST", "测试4: 边界情况处理")
        self.log("TEST", "=" * 60)

        if not self.milvus_ok:
            self.log("SKIP", "Milvus不可达，跳过边界测试")
            self.add_result("边界测试", "SKIP", "Milvus不可达")
            return

        try:
            from rag_qa.core.rag_system import RAGSystem
            rag = RAGSystem()

            test_cases = [
                {
                    "question": "今天天气怎么样",
                    "expected": "refuse",
                    "description": "超范围问题"
                },
                {
                    "question": "你好",
                    "expected": "greeting",
                    "description": "问候语"
                },
                {
                    "question": "a" * 500,
                    "expected": "handle",
                    "description": "超长输入"
                }
            ]

            for case in test_cases:
                question = case["question"]
                expected = case["expected"]
                description = case["description"]

                self.log("INFO", f"\n测试: {description} - '{question[:50]}...'")
                start_time = time.time()

                try:
                    result = rag.generate_answer(
                        query=question,
                        source_filter="management"
                    )
                    duration = time.time() - start_time

                    if isinstance(result, dict):
                        answer = result.get("answer", "")
                    elif isinstance(result, str):
                        answer = result
                    elif isinstance(result, tuple):
                        answer = result[0] if result[0] else ""
                    else:
                        answer = str(result)

                    if expected == "refuse":
                        if "超出" in answer or "无法" in answer or "抱歉" in answer or "人工客服" in answer:
                            self.log("PASS", f"正确拒绝回答 ({duration:.2f}s)")
                            self.add_result(f"边界: {description}", "PASS", "正确拒绝", duration)
                        else:
                            self.log("WARN", f"应该拒绝但没有: {answer[:100]}")
                            self.add_result(f"边界: {description}", "WARN", "未正确拒绝", duration)

                    elif expected == "greeting":
                        if "你好" in answer or "欢迎" in answer or "帮助" in answer:
                            self.log("PASS", f"正确回应问候 ({duration:.2f}s)")
                            self.add_result(f"边界: {description}", "PASS", "正确回应", duration)
                        else:
                            self.log("WARN", f"问候回应不自然: {answer[:100]}")
                            self.add_result(f"边界: {description}", "WARN", "回应不自然", duration)

                    elif expected == "handle":
                        if answer and len(answer) > 0:
                            self.log("PASS", f"成功处理超长输入 ({duration:.2f}s)")
                            self.add_result(f"边界: {description}", "PASS", "成功处理", duration)
                        else:
                            self.log("FAIL", f"超长输入处理失败")
                            self.add_result(f"边界: {description}", "FAIL", "处理失败", duration)

                except Exception as e:
                    duration = time.time() - start_time
                    self.log("FAIL", f"边界测试异常: {str(e)}")
                    self.add_result(f"边界: {description}", "FAIL", str(e), duration)

        except Exception as e:
            self.log("FAIL", f"边界测试初始化失败: {str(e)}")
            self.add_result("边界测试", "FAIL", str(e))

    def print_summary(self):
        total = self.passed + self.failed + self.warnings + self.skipped

        print("\n" + "=" * 70)
        print("📊 冒烟测试总结")
        print("=" * 70)
        print(f"总测试数: {total}")
        print(f"✅ 通过: {self.passed}")
        print(f"⚠️  警告: {self.warnings}")
        print(f"❌ 失败: {self.failed}")
        print(f"⏭️  跳过: {self.skipped}")

        executed = self.passed + self.failed + self.warnings
        if executed > 0:
            pass_rate = self.passed / executed * 100
            print(f"\n通过率: {pass_rate:.1f}% (已执行{executed}个测试)")

        if self.failed > 0:
            print(f"\n❌ 失败的测试:")
            for r in self.results:
                if r["status"] == "FAIL":
                    print(f"  - {r['test']}: {r['details']}")

        if self.warnings > 0:
            print(f"\n⚠️  警告的测试:")
            for r in self.results:
                if r["status"] == "WARN":
                    print(f"  - {r['test']}: {r['details']}")

        if self.skipped > 0:
            print(f"\n⏭️  跳过的测试:")
            for r in self.results:
                if r["status"] == "SKIP":
                    print(f"  - {r['test']}: {r['details']}")

        durations = [r["duration"] for r in self.results if r["duration"] > 0]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            print(f"\n⏱️  性能统计:")
            print(f"  平均响应时间: {avg_duration:.2f}s")
            print(f"  最大响应时间: {max_duration:.2f}s")

        print("=" * 70)

        if self.failed > 0:
            return 2
        elif self.warnings > 0:
            return 1
        else:
            return 0


def main():
    print("\n" + "=" * 70)
    print("🔥 冒烟测试开始")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tester = SmokeTester()
    has_service = tester.check_connectivity()

    if not has_service:
        print("\n❌ 所有服务都不可达，无法执行任何测试！")
        print("请检查 MySQL/Redis/Milvus 服务是否正在运行。")
        sys.exit(2)

    print(f"\n📋 配置信息:")
    print(f"  MySQL: {tester.config.MYSQL_HOST}:{tester.config.MYSQL_PORT}/{tester.config.MYSQL_DB}")
    print(f"  Milvus: {tester.config.MILVUS_HOST}:{tester.config.MILVUS_PORT}/{tester.config.MILVUS_COLLECTION}")
    print(f"  Redis: {tester.config.REDIS_HOST}:{tester.config.REDIS_PORT}/{tester.config.REDIS_DB}")
    print(f"  BM25阈值: {tester.config.BM25_THRESHOLD}")

    tester.test_faq_matching()
    tester.test_rag_retrieval()
    tester.test_end_to_end()
    tester.test_boundary_cases()

    exit_code = tester.print_summary()

    print(f"\n🏁 冒烟测试结束 (退出码: {exit_code})\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
