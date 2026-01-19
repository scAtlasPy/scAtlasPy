
## 📖 项目文档

- [项目介绍](OVERVIEW.md) - scAtlas 概述、核心特性和快速开始指南
- [分支管理说明](#🌿-分支管理说明branch-strategy)

---

## 🌿 分支管理说明（Branch Strategy）

本项目采用 **主干 + 开发 + 个人分支** 的分支管理模式，用于保证代码稳定性与多人协作效率。

---

### 一、分支结构总览

```text
main
 └── dev
      ├── dev-zsp
      └── dev-yyz
````

* **main**：主分支（稳定分支）
* **dev**：开发分支（集成分支）
* **dev-xxx**：个人开发分支（功能开发）

---

### 二、各分支职责说明

#### 1️⃣ main（主分支）

* 存放 **稳定、可运行、可交付** 的代码
* ⚠️不允许直接在 `main` 上开发或提交代码
* 只能通过 **合并 dev 分支** 的方式更新


---

#### 2️⃣ dev（开发分支）

* 用于整合所有成员的阶段性成果
* 所有个人分支的代码，需先合并到 `dev`
* 在 dev 中通过测试后，再合并到 main

---

#### 3️⃣ dev-xxx（个人分支）

* 每位成员使用 **独立的个人分支**
* 命名规范：`dev-姓名缩写 / dev-昵称`

  * 例如：`dev-zsp`、`dev-yyz`
* 所有功能开发、Bug 修复 **只在个人分支完成**

---

### 三、开发流程（标准）

#### 1️⃣ Clone 项目

```bash
git clone https://github.com/scAtlasAnalysis/scAtlas.git
cd scAtlas
```

默认位于 `main` 分支。

---

#### 2️⃣ 切换到个人分支

```bash
git status
git checkout dev-zsp
```



---

#### 3️⃣ 开发 & 提交代码

```bash
# 编写代码
git add .
git commit -m "feat: 实现 xxx 功能"
git push
```

---

#### 4️⃣ 同步 dev 分支的最新代码

在合并前，先同步 `dev`：

```bash
git fetch origin
git rebase origin/dev
```

如有冲突，解决后：

```bash
git rebase --continue
```

---

#### 5️⃣ 合并到 dev 分支

* 通过 Pull Request（推荐）
* 或由负责人手动合并

---

### 四、分支使用规范（重要）

❌ 禁止行为：

* 直接在 `main` 分支提交代码
* 在他人的个人分支上开发
* 未同步 `dev` 就直接合并

✅ 推荐习惯：

* 每天开发前先 `git pull`
* 提交前确认当前分支
* 提交信息清晰、具体

---

### 五、提交信息规范（建议）

```text
feat: 新功能
fix: 修复 bug
docs: 文档修改
refactor: 代码重构
```

示例：

```bash
git commit -m "fix: 修复登录接口参数校验错误"
```

---

### 六、快速自检

```bash
git branch    # 确认当前分支
git status    # 确认工作区状态
```




