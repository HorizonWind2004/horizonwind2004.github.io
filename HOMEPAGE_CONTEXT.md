# 个人主页维护上下文（Ji Xie）

这份文档用于后续快速对齐主页修改风格，避免反复沟通。

## 1) 基本原则

- 页面主文件：`index.html`
- 样式文件：`stylesheet.css`
- 头像资源：`assets/images/`
- 简历文件：`assets/JiXie_CV_2page.pdf`
- 回答与沟通默认中文（简体）

## 2) 个人简介（Profile）约定

- 当前身份写法：现在是 Bytedance Seed intern，之前是 Berkeley BAIR visiting。
- 不写“我是浙大的”身份叙述（学校归属身份不放在简介里）。
- `Rank/GPA` 不放主页正文（放在 CV）。
- 顶部链接包含：`Email / Google Scholar / CV / GitHub / Twitter`。

## 3) Publications 区块约定

- 标题为 `Publications`（不是 `Selected Publications`）。
- 分区顺序固定：
  1. `Foundation Models`（上）
  2. `Application`（下）
- 黄色底（`#ffffd0`）代表 key papers。
- 当前仅两篇黄色：
  - `RECA`
  - `ICEdit`
- 不再额外写 `· Selected`（避免和黄色重复表达）。

## 4) 文案风格约定

- `SSHS` / `VideoCoF` 介绍保持一两句，抓重点，不要过长段落。
- `ICEdit` 允许更宣传化表达（当前是夸张风格）。
- `RECA` 当前固定文案：
  - `Unlocking the massive zero-shot potential in unified multimodal models through self-supervised learning.`
- `3DIS` 需要保留并强调 equal contribution：
  - `(* denotes equal contribution)`（下划线+加粗）

## 5) 作者链接规则

- 能用个人主页就优先个人主页（尤其 VideoCoF）。
- 没有稳定主页时再用 Scholar。
- 实在找不到时可用 arXiv author search 兜底。

当前已确认链接偏好（示例）：

- VideoCoF：
  - Xiangpeng Yang -> `https://xiangpengyang.github.io/`
  - Yiyuan Yang -> Scholar（暂未确认稳定个人主页）
  - Yan Huang -> `https://github.com/Huang-3`
  - Min Xu -> `http://www.uts.edu.au/staff/min.xu`
  - Qiang Wu -> `https://profiles.uts.edu.au/Qiang.Wu`
- SSHS（可继续补全）：
  - Yanhao Jia -> `https://curisejia.github.io/`
  - Hao Li -> `https://scholar.google.com/citations?user=y4va91AAAAAJ&hl=en&oi=ao`
  - Mengmi Zhang -> `https://a0091624.wixsite.com/deepneurocognition-1`

## 6) 奖项（Honors & Awards）约定

- 按年份倒序展示。
- 同一奖项跨年份时合并成一条，避免冗余（如 2024, 2023 放一行）。
- 当前明确：
  - `Zhejiang Provincial Government Scholarship`：2024, 2023 合并
  - `Zhejiang Provincial Collegiate Programming Contest`：2024 & 2023 合并

## 7) 头像切换功能约定

- 默认头像：`assets/images/jp_icon.jpg`（动漫）
- 切换按钮：可切到真实头像 `assets/images/icon.jpg`
- 切到真实头像显示文案：`Taken at BAIR.`
- 按钮历史上出现过“点击无效”问题，后续改动不要破坏以下点：
  - 按钮 `type="button"`
  - 存在可执行的 `toggleAvatar()`（含初始化逻辑）
  - 避免脚本加载时机导致事件未绑定

## 8) 版式与视觉约定

- `Foundation Models` 与 `Application` 间距已额外拉大（中间有 spacer 行）。
- 分区标题（`h3`）做过微调，当前为“小一点点”版本（不要突然放大）。
- 头像区域做过下移处理，避免“头像太靠上”。

## 9) 发布与提交约定

- 用户说 `push` 时，直接执行 `commit + push`。
- 默认不要提交本地编辑器配置（如 `.vscode/`）。
- 提交前建议检查：
  - `git status`
  - `git diff`
  - 页面关键交互（头像按钮）可用性

