# defense_layer

该目录保存 ADM-MRJA 的防御模型核心代码。后续防御框架相关开发默认只在 `C:\Users\hhh\Desktop\毕设\攻击代码复现\ADM-MRJA` 中继续进行。

## 按报告整理后的结构
- `adaptive_defense_framework/module_1_context_state_modeling/`: 模块一，多轮上下文状态建模
- `adaptive_defense_framework/module_2_bidirectional_intent_inference/`: 模块二，双向意图推断
- `adaptive_defense_framework/module_3_adaptive_defense_policy/`: 模块三，自适应防御策略
- `adaptive_defense_framework/shared/`: 三模块共用的支撑代码
- `artifacts/`: 训练后的模型工件

## 兼容说明
- `adaptive_defense_framework/` 根目录仍保留同名文件，作为兼容转发层。
- 现有 `interfaces/` 下脚本无需修改即可继续使用。
- 后续如果继续做模块级开发，优先修改三模块子目录和 `shared/` 中的文件。

## 详细说明
- 总体设计文档：`defense_layer/防御框架详细说明.md`
