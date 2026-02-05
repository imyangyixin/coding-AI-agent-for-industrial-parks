# coding-AI-agent-for-industrial-parks
虚拟产业园AI编码（deepseek）

本项目用于实现一套基于 DeepSeek API 的扎根理论（Grounded Theory）自动编码流程，支持从开放编码到理论叙事的完整链条：

- Module1：开放编码（Open Coding）
- Module2：相关性筛选（Filtering）
- Module3：主轴编码（Axial Coding）
- Module4：选择性编码（Selective Coding）
- Module5：Storyline（理论叙事 + 证据锚点）

所有模块由 `run.py` 统一调度运行，最终输出 Excel / JSON / TXT 结果文件。

请在根目录下创建.env文件并输入：
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions

DEEPSEEK_OPEN_MODEL=deepseek-chat
DEEPSEEK_FILTER_MODEL=deepseek-reasoner
DEEPSEEK_AXIAL_MODEL=deepseek-reasoner
DEEPSEEK_SELECTIVE_MODEL=deepseek-reasoner
DEEPSEEK_STORYLINE_MODEL=deepseek-reasoner
