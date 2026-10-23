# ptx-downgrader

让 CUDA 13 时代的新应用跑在 Pascal(GTX 10 系,sm_61)上的 PTX 降级器。

## 原理

CUDA 13 删除了 sm_61 编译目标,但:
1. 应用二进制(fatbin)里嵌的 **PTX 是架构中立的中间表示**;
2. 驱动(580 legacy 分支)的 **JIT 编译器仍然支持 sm_61**;
3. CUDA 12.8/12.9 的 ptxas 仍能编译 sm_61 目标。

所以路径是:提取应用 PTX → 把 Pascal 不认识的指令(mma/cp.async/mbarrier/TMA 等)
降级为它认识的指令(fma/half2/dp4a/ld.shared+bar.sync)→ 以 sm_61 重新编译 → 注回应用。
全程用户态,不碰驱动签名。

## 里程碑

- **M0 可行性尖刺——已完成(2026-10-11)**:三个降级形态的 kernel
  (f32 vec_add / f16 缩放 / int8 dp4a 点积)在 1070 上经驱动 JIT
  全部 PASS。驱动未动(仍是 472.12)。
  详见 `m0/README.md`。附带发现(已写入降级规则库的雏形):
  1. PTX ISA 7.0 里特殊寄存器(%tid.x 等)不能直接做 add 操作数,须先 mov;
  2. dp4a 不带 .lo/.hi 限定符(与 mul.lo 不同);
  3. mov.f16 不接受立即数,须用 mov.b16。
  本地校验工具:`tools/cuda_nvcc-*/bin/ptxas.exe`(11.4,仅 10MB,无完整工具链)。
- M1 翻译器骨架——**已完成(2026-10-11)**:
  - `downgrader/` 包:解析器(parser.py)→ 规则引擎(engine.py)→ CLI
    (`python -m downgrader 输入.ptx [-o 输出.ptx]`),ptxas 自动校验。
  - 已实现的规则:版本/目标重写(8.x/sm_90+ → 7.0/sm_61)、特殊寄存器物化
    (ISA 8.x 允许 %tid.x 直接进 ALU,7.0 不允许)、mov.f16 立即数改写、
    dp4a.lo 限定符修正、L1::/L2:: 缓存提示剥离、
    **cp.async 同步化降级**(ld.global→st.shared+零填充,commit 丢弃、
    wait→bar.sync)、mbarrier 保守降级(wait→bar.sync,init/arrive 丢弃)。
  - 拒绝列表:wgmma/tcgen05/TMA/cluster 等硬件专属指令,给出人话报错。
  - 验证:`m1/test_e2e.py` 两个"CUDA 13 风格"kernel 在 1070 上全部 PASS
    (fp16/dp4a/cp.async 零填充路径,数值对拍 numpy)。驱动仍是 472.12,未动。
  - 过程发现并修复:标签判定需排除 "::" 操作码;test kernel 的共享内存
    偏移必须用块内 tid(全局索引会越界,与降级器无关)。
- **M1.5 mma 展开已完成(2026-10-12)**:
  - `downgrader/mma.py`:`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    → 共享内存交换(每 lane 存 4 个 A 字 + 2 个 B 字)+ bar.sync +
    每线程 64 项 f32 FMA(4 轮 × 16 项点积)+ D 写回。
    24KB 静态共享缓冲(32 warp × 768B),寄存器池声明每入口注入一次,
    谓词化 mma 拒绝(bar.sync 部分线程死锁)。
  - 布局表逐字取自 PTX ISA 9.0 文档 §"Matrix Fragments for mma.m16n8k16"
    (PDF 已存 tools/ptx_isa.pdf),先由 `tools/sim_mma.py` 纯 Python 模拟
    对拍 numpy(rel err 1e-7)再上 GPU。
  - GPU 实测:max abs err 1.43e-06(浮点结合序噪声),含 A 单位阵探针、
    B 全部 128 个单点探针验证。
  - 调试中修的 bug(全在测试 kernel,降级器本身无误):D 写回误用 C 的
    基址(哨兵值实验实锤)、B 布局三改(列主序→行主序→4B 对齐交错包,
    老式 mov 双标量打包不支持是根因)、b1 偏移 128B 应为 16B。
    排查武器:单点探针逐元素验证映射,比看整体误差快得多。
- M2 golden test + llama.cpp(禁 cuBLAS)端到端;此时才需要把驱动升到 580 legacy
  (提前建系统还原点,保留 472.12 安装包以便回滚)
- **M2 进行中(2026-10-15)**:
  - cuobjdump 11.4 已就位(tools/,6MB);普查脚本 `tools/survey.py`
    (cuobjdump -ptx 提取 → 按 .entry 分段 → 逐 kernel 降级 →
    通过/拒绝统计 + 被拒指令直方图)。
  - 目标二进制:llama.cpp b11574 的 CUDA 13.4 Windows 包(llama-main.zip,
    153MB,内含 ggml-cuda.dll fatbin)。GitHub 直连持续超时(rc=28/35),
    断点续传多轮推进中(45MB/153MB),下载完成后即可普查。
  - **兼容性普查结果(2026-10-16,真实 CUDA 13.4 产物)**:
    cuobjdump 11.4 无法解析新 fatbin(报 sm_16 removed 后中断),
    换 CUDA 13.0 的 cuobjdump 成功,导出 1008 个 kernel 段(896MB 文本)。
    PTX 段为 sm_75/80/90(JIT 回退段)+ sm_86/89/120/121 仅 SASS。
    降级结果(每架构 245 个 kernel):
    - sm_75:208 通过 / 37 拒绝(全是 mma 其它形状)
    - sm_80:208 通过 / 37 拒绝(35 mma 形状 + 2 解析问题)
    - sm_90:83 通过 / 162 拒绝(125 griddepcontrol + 37 mma)
    通过率 **85%(sm_75/80)**。拒绝的三类:
    1. mma 其它形状(m16n8k8 等,35-37 个)→ 下一步实现;
    2. griddepcontrol(125 个,sm_90 的 PDL 启动机制)→ 实际是启动优化,
       保守做法是整 kernel 前两条 no-op 化(语义安全),下一步处理;
    3. 2 个"操作数数量 4"是 fatbin 里 m16n8k16 的变体拼写,待查。
    注:普查是"文本级可降级性",未过 ptxas/GPU 验证;且 k13 产物对
    Pascal 的意义在于:未来 llama.cpp 若发布 CUDA 13-only 构建,
    1070 仍可经"提取 PTX→降级→JIT"续命。
    工具链遗留:cuda_cuobjdump-13.0.85 保留(tools/),大文件已清理。
- **M2.5 规则补全完成(2026-10-17)**:
  - mma 展开从 1 种形状扩到 6 种:m16n8k16/m16n8k8 ×
    (f32.f16.f16.f32 / f16.f16.f16.f16 / f32.bf16.bf16.f32 / f32.tf32.tf32.f32)。
    关键发现(平铺完备性枚举验证):**k8 的 fragment 是 A 2 个 f16x2
    (a0=行 g 列 2t/2t+1,a1=行 g+8 同列)+ B 1 个 f16x2** —— 文档表格里的
    a0..a3/b0 是半字标签而非寄存器名,误读会导致操作数数量对不上
    (k8 f32 acc 实际 11 操作数 = D4+A2+B1+C4)。
  - griddepcontrol(sm_90 PDL 启动提示)剥除:语义安全(只损失启动重叠)。
  - 解析器支持跨行指令拼接(ptxas/cuobjdump 的换行断句)。
  - GPU 验证:m16n8k8 展开 max abs err 9.5e-07(1070 实测)。
    调试教训:测试 kernel 里 A(16B/行)与 C/D(32B/行)行距不同,
    复用基址寄存器导致 C/D 全错——单点探针 + 灰盒 dump 定位。
    另发现 472.12 老 JIT 的寄存器分配 bug:kernel 使用超过 ~13 个 b64
    声明寄存器时 cuModuleLoadDataEx 报 218(JIT log 为空),复用低位
    寄存器号可绕过。此 bug 只影响手写调试 kernel,真实产物经 ptxas
    编译无此问题。
  - **最终普查通过率:91.4%(sm_75)/ 91.9%(sm_80,sm_90)**
    (每架构 221-222 kernel 中 203 通过)。剩余拒绝全部是 int8 mma
    (m16n8k32/m16n8k16/m8n8k16 s32.s8.s8.s32,16-17 个/架构)——
    布局独立,留待下一阶段;以及 2 个疑似解析边缘情况的伪拒绝。
    对比起点(仅 m16n8k16.f32):85% → 91.9%,griddepcontrol 125 个
    全部转为通过。
- **M2.6 真实 kernel 端到端验证完成(2026-10-17)——项目核心闭环达成**:
  - **全程零驱动改动**(472.12 原封未动,应用户要求永久安全优先)。
  - 验证对象:llama.cpp b11574 的 `fwht_cuda_block<1024,256,float>`
    (真实 CUDA 13.4 产物的 sm_75 PTX 段,Fast Walsh-Hadamard 变换)。
  - 三重验证全 PASS:① numpy 参考实现对拍误差 0;② 参照产物
    (原生 PTX 仅重写 .target,指令不动)对拍误差 0;
    ③ **降级全管线产物与参照在 1070 上逐位一致**。
  - 意义:从 CUDA 13.4 二进制提取 → 降级 → Pascal 正确执行 的完整
    链路在真实生态 kernel 上闭环。
  - 期间破案的两个关键机制(重要工程知识,已固化到 verify_fwht.py 注释):
    1. 472.12 老 JIT **只加载 .target sm_61(本机架构)的 PTX 模块**,
       sm_70/sm_75 一律 218。因此"原生参照"不能直接 JIT,必须用 ptxas
       编译成 cubin 后 cuModuleLoadData——这同时证明降级器的
       target 重写不是可选项而是必需步骤。
    2. ptxas 的 -arch 必须与 .target 一致,否则报 fatal(且错误信息
       在 stdout 而非 stderr)。
  - 本版本 fatbin 的观察:105 个唯一 kernel(SSM/MoE/fwht/CUB 工具类),
    主力 matmul/softmax kernel 在仅 SASS 段(无 PTX)—— 对这类 kernel,
    降级器路线不适用,需依赖 llama.cpp 社区重新以 PTX 发布(或自编译
    加 -arch=compute_61)。此为下一步的真实工作项。
  - 驱动升级(580 legacy)被明确推迟:仅在决定跑完整 llama.cpp 程序时
    才考虑,且必须先有:系统还原点 + 472.12 安装包备份 + 安全模式回滚预案。
- **Bug 审查轮(2026-10-17)——发现并修复一个关键正确性 bug**:
  - **[已修复·重要] 混合宽度地址**:PTX 7.0 拒绝 `[64位基址+32位寄存器]`
    形态的地址(实测 ptxas 与老 JIT 都报语法错,特殊寄存器与普通寄存器
    同样适用)。CUDA 13 编译器常态输出这种形态 → 修复:pass_special_regs
    现在对 32 位寄存器偏移统一做 物化 → cvt.u64.u32 → add.s64,返回纯
    [reg64] 地址;纯特殊寄存器操作数与地址内嵌两种形态可并存于同一指令。
    修复经 32-lane 逐 lane 写读一致性实测(0/32 不一致)。
    此前"spike 阶段未暴露"的原因:测试 kernel 恰好都用了立即数偏移或
    先物化的写法。
  - [验证无误] 同一入口 k8+k16+cp.async 混用:__dg_mma 与 %dgmb 声明各
    仅注入一次,ptxas 通过;D=C 原地累加链(两个 mma 共享同一 D/C 寄存器)
    的展开顺序正确(mma1.C读 < mma1.D写 < mma2.C读 < mma2.D写)。
  - [验证无误] 幂等性:全部 4 个测试 kernel 的降级产物再次降级结果不变。
  - [已知限制·记录] mma 共享缓冲按 warp 分区读写,bar.sync 0 为块级,
    无跨 warp race;但若原 kernel 在部分线程分支内调用 mma,降级器已
    显式拒绝(谓词化同理)——此限制已在报错信息中说明。
  - 回归:4 个 e2e 测试全 PASS。
- **深度审查第二轮(2026-10-17)——发现并修复 4 个 bug + 1 个性能级软件降级**:
  1. **[已修·致命] 寄存器池撞名**:%dgmb16<4> 池名与 %dgmb<48> 池的成员
     %dgmb16 撞名,ptxas 报 Illegal expression(f16 累加器路径从未触发过
     是因为此前只测过 f32 acc)。改名 %dgmw<4>,f16 acc 路径 GPU 验证
     输出精确 8.0(全 1 矩阵 × k=8)。
  2. **[已修·死锁] 谓词化 wait → 谓词化 bar.sync**:cp.async.wait_group
     与 mbarrier.wait 被谓词化时,替换产物 bar.sync 继承谓词 → 部分线程
     到不齐整块死锁。修复:替换时剥谓词(wait 本是本线程语义,同步化后
     全员屏障不改变可见性结论)。
  3. **[已修·漏网] ldmatrix/stmatrix/wmma 透传**:与 mma 配套的高频
     sm_75+ 指令此前被透传(老 JIT 必拒)。加入 reject 清单,报错信息
     给出替代写法指引。正确降级 = 重排 fragment 布局,列为 M1.6 候选。
  4. **[已修·透传漏网] bmsk**(sm_100 位域掩码,fatbin 实测 1694 处):
     降级为 shr + 常量掩码 and(掩码在降级期由立即数 len 算出;运行期
     shl 立即数会触发 ptxas 类型推断错误)。GPU 验证 0x3C 语义正确。
  5. **[新增·软件降级] tanh**(sm_75+,fatbin 实测 3271 处,ptxas 直接
     拒绝):降级为 mul/ex2.approx/add/rcp.approx/fma 五条序列
     (tanh(x) = 1 - 2/(1+e^(2x))),f16 变体带 cvt 包裹。
     GPU 验证 max err 5.96e-08(比硬件 approx 更准)。
  - 确认为固有差异非 bug(已在文档):ftz 语义(mma 硬件与 fma.rn 非 ftz
    行为一致)、FMA 累加序(容差内)、f16 acc 的 f32 中间精度(更精确)。
  - 方法论收获:第一轮审查(对抗性手写 kernel)抓到地址形态 bug;
    第二轮(真实 fatbin opcode 审计)抓到 tanh/bmsk——真实产物覆盖的
    指令形态远超手写测试的想象力。tools/audit_opcodes.py 固化此审计。
  - 回归:全部 e2e PASS,磁盘清理完毕。
- M3 通用注入层(Linux LD_PRELOAD / Windows Detours 钩 nvcuda.dll)
- M4 社区化:CI、文档、支持范围声明

## 已知的三堵墙(范围声明)

1. cuBLAS/cuDNN 只发 SASS、无 PTX,降级器碰不了 → 只救自带 kernel 的引擎
   (llama.cpp 可 `-DGGML_CUDA_NO_CUBLAS` 或等效选项)。
2. MMA→FMA 展开会推高寄存器占用,性能预计为原生 50-90%。
3. 生命周期上限约 3-5 年(用户向 Vulkan 后端迁移),但窗口仍在。

## 环境备忘(2026-10-11)

- 测试机:笔记本 GTX 1070 8GB(sm_61)
- 驱动:472.12(R470 分支,CUDA 11.4)——用户特意停留的稳定版,**M0/M1 阶段不动它**
- Python 3.13 + numpy 2.5.3;git 有;无 MSVC、无 CUDA toolkit(有意不装)
- C 盘余 17GB,所有工具选择都按省盘方案走

## v0.2.0 开发记录(2026-10-23)—— 多目标架构支持

- **核心改动**:降级目标从硬编码 sm_61 参数化为 --target sm_61/70/75/80/86/89。
  reject 规则表改为 (引入代, 说明) 结构,按"指令引入代 vs 目标代"判定:
  目标代 >= 引入代的指令原生支持,直接透传。cp.async 降级仅在目标 < sm_80
  时激活。对 30 系(Ampere/sm_86)用户,除 ldmatrix 外的大部分 sm_80 指令
  原生可用,降级管线只处理真正超前的部分。
- CLI 新增 --target;输出文件名后缀随目标变化(.sm86.ptx 等)。
- audit_reject.py 改为按目标报告覆盖率,新增 pending 分类
  (ldmatrix/redux.sync —— 需要 shfl 谓词返回支持,v0.3 计划)。
- 半成品教训:ldmatrix 展开写到一半发现读回寻址需要跨 lane 索引推导,
  与 redux.sync 的 shfl 谓词返回问题同级。诚实做法是删除草稿、只保留
  明确拒绝,README 不写做不到的事。
