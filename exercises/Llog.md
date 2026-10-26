## 准备阶段 环境配置
- 先在wsl2 Ubuntu24环境进行code、本地测试，服务器进行训练。用uv进行包管理。

## Baseline
- 依赖“sympy latex2sympy2-extended math-verify pylatexenc”用于运行drgrpo_grader.py。
- 数据集MATH，7类，选用"prealgebra"进行本地测试，每类5题smoke，然后正式推理。