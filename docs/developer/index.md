# Developer Notes

这一部分主要面向两类读者：希望把自定义算法接入 scAtlasPy 的算法开发者，以及希望修复问题、完善模块的开源贡献者。普通使用者可以先阅读 installation 和 tutorials。

```{toctree}
:maxdepth: 1

data-model
minibatch-architecture
performance
documentation
known-limitations
```

## 推荐阅读顺序

| 目标 | 页面 |
|---|---|
| 理解数据库中保存了哪些表 | {doc}`data-model` |
| 理解 minibatch 底层架构（single-pass / multi-pass / ShuffleBuffer） | {doc}`minibatch-architecture` |
| 调整导入、批读取和分析速度 | {doc}`performance` |
| 修改或新增官网文档 | {doc}`documentation` |
| 查看当前还没有稳定支持的能力 | {doc}`known-limitations` |
