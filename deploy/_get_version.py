import sys, subprocess, pip

print('➡️ 当前python解释器绝对路径:')
python_path = sys.executable
print(python_path)
print('\n\n')

print('➡️ 当前python版本:')
python_version = sys.version
print(python_version)
print('\n\n')

print('➡️ 当前pip版本:')
pip_version = pip.__version__
print(pip_version)
print('\n\n')

print('➡️ 当前python环境三方包安装名及版本信息:')
def get_all_packages():
    # 调用当前python对应的pip freeze
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    pkg_text = result.stdout
    # 打印所有包，每行 包名==版本
    # print(pkg_text)
    # 如需存列表
    pkg_list = pkg_text.strip().splitlines()
    return pkg_list

for item in get_all_packages():
    print(item)
print('\n\n')

print('➡️ 生成requirements.txt文件:')
def export_requirements_file(file_path="a-requirements.txt"):
    # 获取所有依赖包
    pkg_lines = get_all_packages()
    # 写入本地文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pkg_lines))
    print(f"\n成功生成文件：{file_path}")
export_requirements_file()

