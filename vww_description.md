## **5\. 深度方案一：视觉唤醒词检测 (Visual Wake Words)**

### **5.1 项目背景与定义**

“视觉唤醒词”（VWW）是指一种能够检测特定视觉对象（如“人”）是否存在的微型模型。它类似于语音助手中的“Hey Siri”，平时处于超低功耗待机状态，仅在检测到人时唤醒主系统。这是智能门铃、安防摄像头和智能家居的核心技术4。

**项目目标：** 训练一个轻量级的CNN模型，能够以高准确率判断图像中是否存在“人”，且模型参数量适合微控制器部署（\< 250KB）。

### **5.2 数据集策略 (Data Engineering)**

虽然可以使用标准的COCO数据集，但直接下载整个数据集（\>20GB）效率太低。

* **数据源：** 使用tensorflow\_datasets (TFDS) 库直接加载COCO 2017数据。  
* 标签生成逻辑：  
  VWW任务是一个二分类问题（Person vs Not-Person）。COCO本身是多标签检测数据集，需要进行转换：  
  1. 遍历每一张图片。  
  2. 检查标注信息（Annotations）。  
  3. 如果图片包含类别ID为“Person”的对象，**且**该对象的边界框（Bounding Box）面积超过图片总面积的0.5%（微小的人无法作为唤醒信号），则标记为1。  
  4. 否则，标记为0。  
  5. **数据平衡：** COCO中包含人的图片较多，需要对“非人”类别进行过采样或对“人”类别进行欠采样，以保证正负样本比例接近1:115。

### **5.3 模型架构设计：MobileNetV2 Micro**

为了满足嵌入式约束，不能直接使用标准的MobileNetV2。需要调整超参数alpha（宽度乘子）。

**代码逻辑概念：**

Python

import tensorflow as tf

\# 使用预训练权重加速收敛（迁移学习）  
base\_model \= tf.keras.applications.MobileNetV2(  
    input\_shape=(96, 96, 3), \# 输入分辨率降低至96x96，大幅减少计算量  
    alpha=0.35,              \# 宽度乘子设为0.35，模型通道数减少约3倍  
    include\_top=False,       \# 移除顶层的1000类分类器  
    weights='imagenet'       \# 加载ImageNet权重  
)

\# 冻结基座模型，只训练新加的层  
base\_model.trainable \= False

model \= tf.keras.Sequential()

* **输入分辨率：** 96x96是TinyML的黄金标准。它保留了足够的特征用于识别人，同时将RAM需求降至最低。  
* **Alpha=0.35：** 这将模型参数量压缩到几十万级别，生成的TFLite文件仅数百KB。

### **5.4 训练与优化流程 (MLOps)**

1. **迁移学习 (Transfer Learning)：** 使用冻结的基座训练Head层。这只需要几个Epoch即可达到较高精度，且节省时间。  
2. **微调 (Fine-tuning)：** 解冻MobileNetV2的最后几个卷积块，使用极低的学习率（如1e-5）进行微调，以适应“人/非人”的特定特征。  
3. **后训练量化 (Post-Training Quantization, PTQ)：** 这是关键步骤。  
   * 使用tf.lite.TFLiteConverter。  
   * 启用OPTIMIZE\_FOR\_SIZE或DEFAULT优化。  
   * 生成.tflite文件，并对比Float32模型和Int8量化模型的大小与精度17。

### **5.5 评估指标**

* **混淆矩阵 (Confusion Matrix)：** 分析假阳性（False Positive）和假阴性（False Negative）。在唤醒词场景下，假阳性会导致设备频繁唤醒耗电，假阴性会导致用户体验差。  
* **ROC曲线与AUC值：** 评估模型的鲁棒性。  
* **部署指标：** 报告最终的.tflite文件大小（Flash占用）和推理所需的FLOPs。


## **7\. 执行路线图 (10天冲刺计划)**

本计划以\*\*方案A（视觉唤醒词）\*\*为主线，因为其更能体现嵌入式与AI的结合。

| 时间 | 阶段 | 核心任务 | 关键产出 |
| :---- | :---- | :---- | :---- |
| **Day 1** | **准备** | 搭建TensorFlow环境，安装tfds，通读Visual Wake Words论文4。 | 环境配置完成，理解VWW任务定义。 |
| **Day 2** | **数据工程** | 编写脚本从TFDS加载COCO数据，实现过滤逻辑（Person \+ BBox \> 0.5%），制作TFRecord或本地数据集。 | 清洗好的Train/Val数据集，图像Resize至96x96。 |
| **Day 3** | **EDA** | 可视化正负样本，检查图像缩放后的可辨识度。统计样本平衡性。 | 数据探索报告，确认数据质量。 |
| **Day 4** | **模型构建** | 使用Keras搭建MobileNetV2 (alpha=0.35)架构，冻结基座，添加自定义Head。 | 可运行的模型代码。 |
| **Day 5** | **模型训练** | 执行迁移学习（只训练Head），观察Loss曲线。 | 初步训练的模型权重。 |
| **Day 6** | **微调优化** | 解冻部分层，低学习率微调。尝试调整Dropout率以减少过拟合。 | 最终收敛的高精度模型。 |
| **Day 7** | **量化实验** | 使用TFLite Converter进行Int8量化。对比模型大小和精度。 | .tflite文件，量化分析报告。 |
| **Day 8** | **评估分析** | 在测试集上跑推理，计算准确率、混淆矩阵、FPS（估算）。 | 评估图表，性能对比表。 |
| **Day 9** | **报告撰写** | 撰写DTSA 5511期末报告。重点阐述架构选择理由、量化原理及嵌入式部署意义。 | 报告初稿。 |
| **Day 10** | **润色提交** | 检查格式，完善GitHub Readme（作为作品集），提交作业。 | 最终交付物。 |

---

## **8\. 结论**

对于一位即将获得数据科学硕士学位的资深嵌入式工程师而言，您的职业天花板远高于传统的固件开发或普通的数据分析。**Edge AI** 是硬件限制与算法智能碰撞的火花点，也是您背景优势最大化的领域。

通过执行**视觉唤醒词（Visual Wake Words）项目，您不仅能够出色地完成DTSA 5511的课程要求，更能构建一个极具说服力的技术作品集。该项目涵盖了数据清洗、深度学习建模（CNN）、迁移学习以及对于嵌入式工程师至关重要的模型压缩与量化**技术。这一经历将直接为您向**边缘AI架构师**或**AI系统工程师**等高薪职位转型奠定坚实基础。

建议立即着手项目A的实施，利用您对底层资源的敏感度，训练出一个既精准又极致轻量的模型，向未来的雇主展示“软硬结合”的强大威力。


---
---




### 1. 项目阶段规划 (Roadmap)

为了在 **1-2周** 内完成从训练到嵌入式部署的全流程，我建议将项目分为 **4 个阶段 (Phases)**。我们目前处于 Phase 1 和 Phase 2 的过渡期。

*   **Phase 1: 环境搭建与基准验证 (已完成)**
    *   **目标：** 跑通数据加载、预处理、模型构建流程。
    *   **产出：** 确认代码无 Bug，模型结构正确，参数量符合预期。
*   **Phase 2: 真实训练与模型评估 (当前阶段)**
    *   **目标：** 获取真实数据，训练 MobileNetV2 和 V3，对比 Float32 模型的准确率。
    *   **核心任务：**
        *   解决数据集问题（见下文）。
        *   实施迁移学习 (Transfer Learning)。
        *   绘制 Loss/Accuracy 曲线，评估过拟合情况。
        *   **产出：** 两个训练好的 `.keras` 或 `.h5` 模型文件，准确率达到基准线（如 >85%）。
*   **Phase 3: 量化与嵌入式优化 (Edge AI 核心)**
    *   **目标：** 将模型压缩到 KB 级别，模拟嵌入式环境。
    *   **核心任务：**
        *   **TFLite 转换：** Float32 -> Int8。
        *   **PTQ (Post-Training Quantization)：** 制作 Representative Dataset 进行全整型量化。
        *   **对比分析：** 比较量化前后的 Size (KB) 和 Accuracy 损失。
    *   **产出：** `.tflite` 文件，量化分析报告。
*   **Phase 4: 仿真部署与最终报告**
    *   **目标：** 验证模型在“端侧”的运行效果。
    *   **核心任务：**
        *   使用 `tf.lite.Interpreter` 在 Python 中模拟 MCU 推理过程。
        *   测试单张图片的推理耗时 (Latency)。
        *   整理所有图表（混淆矩阵、ROC、参数对比、量化对比），撰写报告。

---

### 2. 数据集解决方案 (Dataset Strategy)

你遇到的问题非常现实：**COCO 太大 (20GB)，Wake Vision 更大 (200GB)，而我们需要的是一个“小而美”的子集。**

在 Colab 上下载 20GB 数据不仅慢，而且解压和处理极易导致 Session 崩溃。对于学习和演示项目，我们不需要工业级的海量数据。

**我的推荐方案：使用 "INRIA Person Dataset" 或 Kaggle 上的精简版**

这是一个经典的行人检测数据集，非常适合 VWW 这种二分类任务。
*   **优点：**
    *   **体积适中：** 约 **970MB**（压缩包更小），Colab 下载只需几分钟。
    *   **结构清晰：** 已经分好了 `pos` (有人) 和 `neg` (无人) 文件夹，几乎不需要清洗。
    *   **质量高：** 图片清晰度高，背景多样。

**操作步骤 (将在 Phase 2 代码中实现)：**

1.  **直接下载：** 在 Colab 中使用 `wget` 命令直接下载 INRIA 数据集的压缩包。
2.  **解压与重组：** 编写脚本解压，并将图片移动到我们 Phase 1 定义好的 `vww_dataset/train/person` 和 `vww_dataset/train/not_person` 结构中。
3.  **无缝衔接：** 这样你 Phase 1 写好的 `image_dataset_from_directory` 代码完全不用改，直接就能读到真实数据了！

---

### 下一步行动

如果你同意这个方案，我将为你生成 **Phase 2 的 Colab 代码**。

**Phase 2 代码将包含：**
1.  **自动下载脚本：** 自动下载并解压 INRIA Person 数据集。
2.  **数据重组脚本：** 将其转换为 Keras 友好的文件夹结构。
3.  **训练循环 (Training Loop)：** 包含 EarlyStopping 和 ModelCheckpoint 回调。
4.  **训练可视化：** 绘制 V2 和 V3 的训练曲线对比。

**你是否同意使用 INRIA 数据集作为替代方案？** (这能为你节省至少 2-3 天的数据处理时间)


----

### 关键词
* 诸如**深度可分离卷积**、**反向残差**和**注意力机制**等架构创新使得在不牺牲准确性的前提下压缩模型复杂度成为可能。
* 一系列优化技术，包括**量化感知训练（QAT）**、**结构化剪枝**、**知识蒸馏**和**低秩分解**，显著降低了深度模型的运行时间和内存需求 。
* ，引入了**神经架构搜索（NAS）**框架（例如MCUNet和TinyNAS），这些框架能够协同优化模型拓扑和部署约束，已经证明ImageNet规模的任务可以在仅配备480 kB SRAM的MCU上执行 。
* new developments in **on-device** and **continual learning** allow models to adapt in real-time under strict memory and compute constraints, further extending the practicality of TinyDL systems

---

## Tiny ML and Tiny DL 模型评测指标和方法


### 1. 四大核心评测指标 (The Big Four Metrics)

#### A. 精度 (Accuracy / Quality)
这是底线。如果模型预测不对，再快也没用。
* **指标：** Top-1 Accuracy (分类), mAP (目标检测), F1-Score (异常检测)。
* **TinyML 特点：** 我们通常追求**“足够好” (Good Enough)** 而不是 SOTA。
    * *例子：* ResNet-50 在 ImageNet 上有 76% 的精度，MobileNetV1 可能只有 70%。但在 MCU 上，为了能跑起来，我们愿意接受这 6% 的损失。

#### B. 延迟 (Latency / Inference Time)
这是实时性的关键。
* **定义：** 处理**单次**推理（Inference）所需的时间（ms）。
* **指标：** `ms per inference`。
* **TinyML 特点：** 这直接决定了应用场景。
    * *电机故障检测：* 需要 < 10ms (为了快速停机)。
    * *VWW (人检测)：* < 200ms 就够了（人走得慢）。

#### C. 能耗 (Energy / Power Consumption) —— *最硬核的指标*
这是嵌入式设备的命门。
* **定义：** 完成一次推理消耗的能量。
* **指标：** **微焦耳 ($\mu J$ / inference)** 或 **毫瓦 (mW)**。
* **换算：** `Energy (J) = Power (W) × Time (s)`。
* **TinyML 特点：** 我们不仅看推理时的功耗，还要看**空闲/睡眠功耗**。如果你的 NPU 很快，但待机漏电严重，那也是不合格的。

#### D. 内存占用 (Memory Footprint) —— *生死的门槛*
这是决定模型“能不能装进去”的硬指标。你必须区分两种内存：
1.  **Flash (Read-Only / NVM):** 存放**模型权重 (Weights)** 和 **代码 (Code)**。
    * *评测点：* 模型文件大小 (Model Size)。
2.  **SRAM (Read-Write / Volatile):** 存放**激活值 (Activations)**、**输入/输出缓冲**和**中间变量**。
    * *评测点：* **Peak RAM Usage (峰值内存)**。这是最大的瓶颈！很多模型权重只有 100KB，但中间层的 Feature Map 需要 500KB RAM，这在只有 256KB RAM 的 MCU 上直接 OOM (Out of Memory)。

---

### 2. 评测方法论 (Methodologies)

如何测量上述指标？行业内分为三个层级：

#### 层级 1：理论计算 (Theoretical Counting) —— *看纸面数据*
* **方法：** 计算模型的 **MACs** (Multiply-Accumulate Operations，乘加运算数) 或 **FLOPs** (浮点运算数)。
* **用途：** 快速估算模型复杂度。
* **陷阱：** **FLOPs $\neq$ Latency**。
    * 在 MCU 上，**内存访问 (Memory Access)** 往往比计算更耗时。一个 FLOPs 很低但内存读写频繁的模型（如 ShuffleNet），在某些 MCU 上可能比 FLOPs 高但结构规整的模型（如 MobileNet）跑得更慢。

#### 层级 2：软件仿真 (Software Simulation) —— *开发阶段*
* **方法：** 使用指令集模拟器 (ISS)，如 Arm 的 **FVP (Fixed Virtual Platforms)** 或 **Renode**。
* **用途：** 在没有硬件板子时，通过 Profiling 工具查看具体的 Cycle Count (时钟周期数) 和 内存峰值。
* **工具：** **STM32Cube.AI** 的 "Analyze" 功能极其强大，它能静态分析出模型需要的确切 Flash 和 RAM 大小。

#### 层级 3：硬件在环 (Hardware-in-the-Loop, HIL) —— *最终验收*
这是最真实、也是 MLPerf 采用的标准方法。

1.  **测延迟：**
    * **GPIO 法（最准）：** 推理开始前拉高一个 GPIO 引脚，推理结束后拉低。用**示波器**测量高电平持续时间。
    * **SysTick 法：** 在代码里读取 MCU 的系统时钟计数器。
2.  **测能耗：**
    * 使用**高精度电源分析仪**（如 **Nordic PPK2** 或 Joulescope）。
    * 将其串联在开发板的电源输入端，以高采样率（如 100kHz）记录电流波形，然后积分计算能量。

---

### 3. 综合评估图表：帕累托前沿 (Pareto Frontier)

在 TinyML 论文或技术报告中，你不能只列一个表格。最专业的展示方式是画 **Accuracy-Latency Trade-off 曲线**（帕累托前沿）。



* **X 轴：** 延迟 (Latency) 或 模型大小 (Model Size) —— *越小越好*。
* **Y 轴：** 准确率 (Accuracy) —— *越高越好*。
* **点：** 每一个点代表一个模型（或同一个模型的不同量化版本）。
* **前沿 (Frontier)：** 位于最左上角的那些点，代表了当前的**最优解**（在同等延迟下精度最高，或同等精度下延迟最低）。

