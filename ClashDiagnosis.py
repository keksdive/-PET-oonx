import subprocess
import re
import sys
import ctypes
import platform
from typing import Optional, Dict

# 检查是否为 Windows 系统
if platform.system() != "Windows":
    print("❌ 此脚本仅支持 Windows 系统！")
    sys.exit(1)

# 检查是否以管理员身份运行
def is_admin():
    """检查脚本是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_command(cmd: str, encoding: str = "gbk") -> Dict[str, any]:
    """
    执行系统命令并返回结果
    :param cmd: 要执行的命令
    :param encoding: 命令输出的编码格式（Windows 默认 gbk）
    :return: 包含返回码、标准输出、标准错误的字典
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        return {
            "code": result.returncode,
            "stdout": result.stdout.decode(encoding, errors="ignore").strip(),
            "stderr": result.stderr.decode(encoding, errors="ignore").strip()
        }
    except subprocess.TimeoutExpired:
        return {"code": -1, "stdout": "", "stderr": "命令执行超时"}
    except Exception as e:
        return {"code": -2, "stdout": "", "stderr": f"执行失败: {str(e)}"}

def check_clash_process() -> str:
    """检查 Clash 进程状态"""
    print("\n[1/8] 🔍 检查 Clash 进程状态")
    cmd_result = run_command("tasklist | findstr /i Clash")
    if cmd_result["code"] == 0 and cmd_result["stdout"]:
        return "✅ Clash 进程正在运行"
    else:
        return "❌ Clash 进程未启动/已崩溃/未检测到"

def check_tun_driver() -> str:
    """检查 TUN/TAP 驱动状态"""
    print("\n[2/8] 🔍 检查 TUN/TAP 驱动状态")
    cmd_result = run_command("sc query tap0901")
    if cmd_result["code"] == 0 and "RUNNING" in cmd_result["stdout"]:
        return "✅ TAP 驱动已安装并正常运行"
    elif cmd_result["code"] == 1060:  # 驱动未安装
        return "❌ TAP 驱动未安装，请在 Clash 设置中重新安装 TUN 驱动"
    else:
        return f"⚠️ TAP 驱动状态异常: {cmd_result['stderr']}"

def check_clash_ports() -> str:
    """检查 Clash 默认端口占用（7890/7891/7892/9090）"""
    print("\n[3/8] 🔍 检查 Clash 默认端口占用")
    ports = [7890, 7891, 7892, 9090]
    result = []
    cmd_result = run_command("netstat -ano | findstr /i LISTENING")
    if cmd_result["code"] != 0:
        return "❌ 无法获取端口信息"
    
    for port in ports:
        if f":{port}" in cmd_result["stdout"]:
            # 提取占用端口的 PID
            pid = re.findall(f":{port}.*LISTENING.*?(\d+)", cmd_result["stdout"])
            result.append(f"✅ 端口 {port} 已被占用 (PID: {pid[0] if pid else '未知'})")
        else:
            result.append(f"❌ 端口 {port} 未被占用（Clash 可能未正常监听）")
    return "\n    ".join(result)

def check_dns_pollution() -> str:
    """检测 DNS 污染（对比国内/海外 DNS 解析结果）"""
    print("\n[4/8] 🔍 检测 DNS 污染（以 youtube.com 为例）")
    # 国内 DNS：阿里云 223.5.5.5
    cn_dns_result = run_command("nslookup youtube.com 223.5.5.5")
    # 海外 DNS：Google 8.8.8.8
    us_dns_result = run_command("nslookup youtube.com 8.8.8.8")
    
    # 提取解析的 IP 地址
    def extract_ip(nslookup_output: str) -> Optional[str]:
        ip_pattern = re.compile(r'Address: (\d+\.\d+\.\d+\.\d+)')
        matches = ip_pattern.findall(nslookup_output)
        return matches[-1] if matches else None
    
    cn_ip = extract_ip(cn_dns_result["stdout"])
    us_ip = extract_ip(us_dns_result["stdout"])
    
    if not cn_ip or not us_ip:
        return "⚠️ 无法获取 DNS 解析结果，可能网络异常"
    elif cn_ip == us_ip:
        return f"✅ DNS 解析结果一致，未检测到明显污染\n    国内 DNS 解析: {cn_ip}\n    海外 DNS 解析: {us_ip}"
    else:
        return f"❌ DNS 解析结果不一致，存在污染风险\n    国内 DNS 解析: {cn_ip}\n    海外 DNS 解析: {us_ip}"

def check_node_connectivity() -> str:
    """测试典型节点域名连通性（以 v2alinodecc.com:23330 为例）"""
    print("\n[5/8] 🔍 测试节点域名连通性")
    # 使用 PowerShell 的 Test-NetConnection 测试端口连通性
    cmd = 'powershell -Command "Test-NetConnection v2alinodecc.com -Port 23330 | Select-Object TcpTestSucceeded"'
    cmd_result = run_command(cmd, encoding="utf-8")
    if "True" in cmd_result["stdout"]:
        return "✅ 节点域名 + 端口 可连通"
    elif "False" in cmd_result["stdout"]:
        return "❌ 节点域名 + 端口 无法连通（节点失效/端口被封）"
    else:
        return f"⚠️ 测试失败: {cmd_result['stderr']}"

def check_system_proxy() -> str:
    """检查系统代理设置"""
    print("\n[6/8] 🔍 检查系统代理设置")
    cmd_result = run_command('reg query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings" | findstr /i ProxyEnable')
    if cmd_result["code"] == 0 and "0x1" in cmd_result["stdout"]:
        return "✅ 系统代理已开启"
    elif cmd_result["code"] == 0 and "0x0" in cmd_result["stdout"]:
        return "❌ 系统代理未开启，请检查 Clash 的「系统代理」开关"
    else:
        return "⚠️ 无法读取系统代理设置"

def check_winsock() -> str:
    """检测 Winsock 状态"""
    print("\n[7/8] 🔍 检测 Winsock 状态")
    cmd_result = run_command("netsh winsock show catalog | findstr /i Clash")
    if cmd_result["code"] == 0 and cmd_result["stdout"]:
        return "✅ Clash 已注入 Winsock 层"
    else:
        return "❌ Clash 未注入 Winsock 层，建议重置网络栈"

def repair_network_stack() -> str:
    """一键修复网络栈（重置 Winsock + IP 配置）"""
    print("\n[8/8] 🔧 一键修复网络栈（可选）")
    choice = input("是否执行 Winsock + IP 配置重置？(y/n): ").strip().lower()
    if choice != "y":
        return "❌ 已跳过修复操作"
    
    # 重置 Winsock
    winsock_result = run_command("netsh winsock reset")
    # 重置 IP 配置
    ip_reset_result = run_command("netsh int ip reset")
    
    if winsock_result["code"] == 0 and ip_reset_result["code"] == 0:
        return "✅ 网络栈重置完成！请重启电脑后重新测试 Clash"
    else:
        return f"⚠️ 修复失败\n    Winsock 重置: {winsock_result['stderr']}\n    IP 重置: {ip_reset_result['stderr']}"

def main():
    """主函数：执行所有诊断步骤"""
    print("="*60)
    print("🛠️  Clash 深度故障诊断脚本（Python 版）")
    print("="*60)
    
    # 检查管理员权限
    if not is_admin():
        print("❌ 请以【管理员身份】运行此脚本！")
        sys.exit(1)
    
    # 执行所有诊断步骤
    diagnostics = [
        check_clash_process(),
        check_tun_driver(),
        check_clash_ports(),
        check_dns_pollution(),
        check_node_connectivity(),
        check_system_proxy(),
        check_winsock(),
        repair_network_stack()
    ]
    
    # 输出最终诊断报告
    print("\n" + "="*60)
    print("📊 最终诊断报告")
    print("="*60)
    for diag in diagnostics:
        print(diag)
    print("="*60)
    input("\n诊断完成，按回车键退出...")

if __name__ == "__main__":
    main()