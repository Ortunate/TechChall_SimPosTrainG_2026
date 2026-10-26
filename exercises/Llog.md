## 准备阶段 环境配置
- 先在wsl2 Ubuntu24环境进行code、本地测试，服务器进行训练。用uv进行包管理。

## Baseline
- 依赖“sympy latex2sympy2-extended math-verify pylatexenc”用于运行drgrpo_grader.py。
- 数据集MATH，7类，选用"prealgebra"进行本地测试，每类5题smoke，然后正式推理。
- 第一次连接服务器，torch、cuda依赖下载由于之前在wsl上用uv锁定了官方下载源，极慢；更换清华镜像，仍慢；切换南大镜像后正常。
- 采用AI写的脚本一键配置环境，其中包含两个独立的python虚拟环境，train与vLLM。
```bash scripts/setup_env.sh \[all || train || vllm]```